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
            self.modelName = 'resnet'

        elif arch.startswith('vgg'):
            # Feature extraction layer
            self.features = original_model.features
            
            # Using AdativeAveragePooling for 448x448 images
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

            # Classifier with classes
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

            # Model Name
            self.modelName = 'vgg'

        elif arch.startswith('efficientnet_b0'):

            # Feature extraction layer
            self.features = original_model.features

            # Using AdativeAveragePooling for 448x448 images
            self.avgpool = original_model.avgpool

            # Classifier with classes
            self.classifier = nn.Sequential(
                nn.Linear(1280, num_classes),
            )

            # Model Name
            self.modelName = 'efficientnet_b0'

        elif arch.startswith('efficientnet_b4'):
            # Feature extraction layer
            self.features = original_model.features

            # Using AdativeAveragePooling for 448x448 images
            self.avgpool = original_model.avgpool

            # Classifier with classes
            self.classifier = nn.Sequential(
                nn.Linear(1792, num_classes),
            )

            # Model Name
            self.modelName = 'efficientnet_b4'

        else:
            raise "Finetuning not supported on this architecture !!"

        # freeze the weight the feature extraction layers
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        # Pass feature extraction layer
        f = self.features(x)
        
        if self.modelName == 'vgg' or self.modelName == 'efficientnet_b0' or self.modelName == 'efficientnet_b4':
            f = self.avgpool(f)

        # Flatten the layer before enter to FCN layer
        f = f.view(f.size(0), -1)
        # Pass FCN Classifier layer
        y = self.classifier(f)

        return y
