import matplotlib.pyplot as plt
import torch
import torchhd 

class Visualizer:
    def __init__(self, encoder, model):
        self.encoder = encoder
        self.model = model
        self.device = next(encoder.parameters()).device

    def get_importance_map(self, sample_digit):
        """Calculate importance map for a digit"""
        with torch.no_grad():
            orig_encoding = self.encoder(sample_digit.unsqueeze(0))
            orig_output = self.model(orig_encoding, dot=True)
            predicted_class = orig_output.argmax(dim=1)
            
            class_prototype = self.model.weight[predicted_class].squeeze()
            importance = torch.zeros((28, 28), device=self.device)
            
            flat_img = sample_digit.flatten()
            position_encodings = self.encoder.position.weight
            value_encodings = self.encoder.value(flat_img)
            
            for pos in range(28*28):
                pos_encoding = position_encodings[pos].unsqueeze(0)
                val_encoding = value_encodings[pos].unsqueeze(0)
                contribution = torchhd.bind(pos_encoding, val_encoding)
                
                similarity = torch.cosine_similarity(
                    contribution, 
                    class_prototype.unsqueeze(0),
                    dim=1
                )
                importance[pos//28, pos%28] = similarity.item()
                
            importance = (importance - importance.min()) / (importance.max() - importance.min())
            return importance, predicted_class.item()

    def visualize_multiple_digits(self, samples, labels):
        """Visualize multiple digits in a 3x3 grid"""
        fig = plt.figure(figsize=(15, 15))
        
        for idx in range(9):
            # Original digit subplot
            ax1 = plt.subplot(3, 6, idx*2 + 1)
            sample_digit = samples[idx].to(self.device)
            ax1.imshow(sample_digit.cpu()[0], cmap='gray')
            
            # Calculate importance map
            importance, pred_class = self.get_importance_map(sample_digit)
            
            ax1.set_title(f'Digit {labels[idx]}\nPred: {pred_class}')
            ax1.axis('off')
            
            # Importance map subplot
            ax2 = plt.subplot(3, 6, idx*2 + 2)
            im = ax2.imshow(importance.cpu(), cmap='hot')
            ax2.set_title('HD Space Contribution')
            ax2.axis('off')
            
            # Add colorbar for each importance map
            plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.show()

