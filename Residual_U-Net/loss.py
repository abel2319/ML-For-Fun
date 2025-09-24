import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss
    """
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, sigma, channels))
    
    def _gaussian(self, window_size, sigma):
        # Create gaussian kernel more efficiently
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()
    
    def _create_window(self, window_size, sigma, channels):
        _1D_window = self._gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channels) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()

class EdgeLoss(nn.Module):
    """
    Edge preservation loss using Sobel operators
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def forward(self, output, target):
        # Calculate gradients
        grad_output_x = F.conv2d(output, self.sobel_x, padding=1, groups=3)
        grad_output_y = F.conv2d(output, self.sobel_y, padding=1, groups=3)
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        
        # Edge magnitude
        edge_output = torch.sqrt(grad_output_x**2 + grad_output_y**2 + 1e-8)
        edge_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-8)
        
        return F.l1_loss(edge_output, edge_target)

class EnhancedDenoisingLoss(nn.Module):
    """
    Enhanced loss function for Monte Carlo denoising with structural similarity
    """
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, ssim_weight=0.3, edge_weight=0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        
        # VGG for perceptual loss
        if perceptual_weight > 0:
            vgg = models.vgg16(pretrained=True).features[:16]
            self.vgg = vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        else:
            self.vgg = None
            
        # SSIM loss
        if ssim_weight > 0:
            self.ssim_loss = SSIMLoss(channels=3)
        else:
            self.ssim_loss = None
            
        # Edge loss
        if edge_weight > 0:
            self.edge_loss = EdgeLoss()
        else:
            self.edge_loss = None
    
    def forward(self, output, target):
        total_loss = 0.0
        loss_components = {}
        
        # L1 Loss (pixel-wise reconstruction)
        if self.l1_weight > 0:
            l1_loss = F.l1_loss(output, target)
            total_loss += self.l1_weight * l1_loss
            loss_components['l1'] = l1_loss.item()
        
        # Perceptual Loss (high-level features)
        if self.perceptual_weight > 0 and self.vgg is not None:
            output_features = self.vgg(output)
            target_features = self.vgg(target)
            perceptual_loss = F.mse_loss(output_features, target_features)
            total_loss += self.perceptual_weight * perceptual_loss
            loss_components['perceptual'] = perceptual_loss.item()
        
        # SSIM Loss (structural similarity)
        if self.ssim_weight > 0 and self.ssim_loss is not None:
            ssim_loss = self.ssim_loss(output, target)
            total_loss += self.ssim_weight * ssim_loss
            loss_components['ssim'] = ssim_loss.item()
        
        # Edge Loss (edge preservation)
        if self.edge_weight > 0 and self.edge_loss is not None:
            edge_loss = self.edge_loss(output, target)
            total_loss += self.edge_weight * edge_loss
            loss_components['edge'] = edge_loss.item()
        
        return total_loss, loss_components

# Alternative: Simple SSIM + L1 combination for better performance
class SimpleDenoisingLoss(nn.Module):
    """
    Lightweight version with just L1 + SSIM for structural similarity
    """
    def __init__(self, l1_weight=0.7, ssim_weight=0.3):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.ssim_loss = SSIMLoss(channels=3)
    
    def forward(self, output, target):
        l1_loss = F.l1_loss(output, target)
        ssim_loss = self.ssim_loss(output, target)
        
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        return total_loss

# Usage examples:
if __name__ == "__main__":
    # Example 1: Full enhanced loss
    criterion_full = EnhancedDenoisingLoss(
        l1_weight=1.0,      # Pixel accuracy
        perceptual_weight=0.1,  # Perceptual quality
        ssim_weight=0.3,    # Structural similarity
        edge_weight=0.2     # Edge preservation
    )
    
    # Example 2: Lightweight SSIM + L1
    criterion_simple = SimpleDenoisingLoss(
        l1_weight=0.7,      # Pixel accuracy
        ssim_weight=0.3     # Structural similarity
    )
    
    print("Enhanced denoising losses ready!")
    print("Use EnhancedDenoisingLoss for maximum quality")
    print("Use SimpleDenoisingLoss for better performance")