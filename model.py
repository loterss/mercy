import torch.nn as nn


# Create model for training
class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes, freeze=True):
        super(FineTuneModel, self).__init__()
        """
        Create Fine-Tune Model for Resnet 50 and VGG16
          Args:
            original_model: torch vision - transfer learning model (expected to be Resnet and VGG model)
            arch: ['resnet', 'vgg']
            num_classes: number of classification classes
            freeze: freezing weight of feature extraction layer
        """

        if arch.startswith('resnet'):
            # Feature extraction layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])

            # Classifier with classes
            self.classifier = nn.Sequential(
                nn.Linear(2048, num_classes)
            )

            # Model Name
            self.modelName = 'resnet50'

        elif arch.startswith('vgg'):
            # Feature extraction layer
            self.features = original_model.features

            # Classifier with classes
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool(output_size=(7,7)),
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

            # Model Name
            self.modelName = 'vgg16'
        else:
            raise "Finetuning not supported on this architecture !!"

        # freeze the weight the feature extraction layers
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        # Pass feature extraction layer
        f = self.features(x)
        # Flatten the layer before enter to FCN layer
        f = f.view(f.size(0), -1)
        # Pass FCN Classifier layer
        y = self.classifier(f)

        return y
