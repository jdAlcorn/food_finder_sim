class RLVisionNetwork(nn.Module):
    """
    Actor-Critic network with 1D CNN vision encoder + MLP fusion.
    Designed for egocentric ray-based vision.
    """

    def __init__(
        self,
        num_rays: int = 128,
        num_channels: int = 3,
        proprio_dim: int = 4,
        init_seed: int = None
    ):
        super().__init__()

        if init_seed is not None:
            torch.manual_seed(init_seed)

        self.num_rays = num_rays
        self.num_channels = num_channels
        self.proprio_dim = proprio_dim

        # ---- Vision encoder (spatial inductive bias lives here) ----
        self.vision_encoder = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)   # collapse ray dimension
        )

        vision_feat_dim = 64

        # ---- Fusion + control ----
        self.fusion_mlp = nn.Sequential(
            nn.Linear(vision_feat_dim + proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(128, 2)
        self.critic_head = nn.Linear(128, 1)

        # Gaussian exploration
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Split observation
        vision = obs[:, :-self.proprio_dim]      # [B, 384]
        proprio = obs[:, -self.proprio_dim:]     # [B, 4]

        # Reshape vision -> [B, C, R]
        vision = vision.view(batch_size, self.num_channels, self.num_rays)

        # Encode vision
        vision_feat = self.vision_encoder(vision).squeeze(-1)  # [B, 64]

        # Fuse
        fused = torch.cat([vision_feat, proprio], dim=-1)
        encoded = self.fusion_mlp(fused)

        # Outputs
        action_logits = self.actor_head(encoded)
        value = self.critic_head(encoded).squeeze(-1)

        log_std = self.log_std.expand(batch_size, -1)
        log_std = torch.clamp(log_std, min=MIN_LOG_STD)

        return action_logits, log_std, value

class RLMLPPolicy(Policy):
    """
    RL-based MLP policy that implements the Policy interface
    Uses Actor-Critic with MLP for continuous action control (no recurrence)
    """
    
    def __init__(self, encoder_dims: Tuple[int, ...] = (256, 128), device: str = "cpu", 
                 v_scale: float = 400.0, omega_scale: float = 10.0, init_seed: int = None):
        """
        Initialize RL MLP policy
        
        Args:
            encoder_dims: MLP encoder hidden layer dimensions
            device: Device to run model on ("cpu" or "cuda")
            v_scale: Velocity normalization scale
            omega_scale: Angular velocity normalization scale
            init_seed: Random seed for deterministic initialization
        """
        self.name = "RLMLP"
        self.encoder_dims = encoder_dims
        self.v_scale = v_scale
        self.omega_scale = omega_scale
        self.init_seed = init_seed
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Create network
        self.network = RLMLPNetwork(
            input_dim=388,
            encoder_dims=encoder_dims,
            init_seed=init_seed
        ).to(self.device)
        
        # Episode step counter (for compatibility)
        self._episode_step = 0
    
    def reset(self) -> None:
        """Reset policy state for new episode"""
        self._episode_step = 0
    
    def act(self, sim_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get action from current simulation state (Policy interface)
        
        Args:
            sim_state: Dictionary containing simulation state
            
        Returns:
            Dict with 'steer' and 'throttle' keys
        """
        # Build observation from sim_state
        observation = build_observation(sim_state, self.v_scale, self.omega_scale)
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get action (deterministic for evaluation)
        with torch.no_grad():
            action, _, _ = self.network.act(obs_tensor, deterministic=True)
        
        self._episode_step += 1
        
        # Convert to numpy and extract single values
        action_np = action.cpu().numpy()[0]
        throttle_val = float(action_np[0])
        steer_val = float(action_np[1])
        
        # Safety clamps
        throttle_val = np.clip(throttle_val, 0.0, 1.0)
        steer_val = np.clip(steer_val, -1.0, 1.0)
        
        return {'throttle': throttle_val, 'steer': steer_val}
    
    def act_training(self, sim_state: Dict[str, Any], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action with training data (log_prob, value) from simulation state
        
        Args:
            sim_state: Dictionary containing simulation state
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action_tensor, log_prob, value)
        """
        # Build observation from sim_state
        observation = build_observation(sim_state, self.v_scale, self.omega_scale)
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get action with training data
        action, log_prob, value = self.network.act(obs_tensor, deterministic=deterministic)
        
        self._episode_step += 1
        
        return action, log_prob, value
    
    def act_batch(self, observations: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch action prediction for training
        
        Args:
            observations: Observation batch [batch_size, 388]
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, log_probs, values)
        """
        return self.network.act(observations, deterministic)
    
    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Get value estimates for observations"""
        _, _, values = self.network.forward(observations)
        return values
    
    def save_weights(self, path: str):
        """Save model weights to file"""
        torch.save(self.network.state_dict(), path)
    
    def load_weights(self, path: str):
        """Load model weights from file"""
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
    
    def get_params(self) -> Dict[str, Any]:
        """Get policy parameters for serialization"""
        return {
            'encoder_dims': list(self.encoder_dims),
            'device': str(self.device),
            'v_scale': self.v_scale,
            'omega_scale': self.omega_scale,
            'init_seed': self.init_seed
        }
    
    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> 'RLMLPPolicy':
        """Create policy from parameters dict"""
        return cls(
            encoder_dims=tuple(params['encoder_dims']),
            device=params['device'],
            v_scale=params['v_scale'],
            omega_scale=params['omega_scale'],
            init_seed=params['init_seed']
        )