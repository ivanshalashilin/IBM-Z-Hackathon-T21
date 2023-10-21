# AI classification of jellyfish and plastic from ocean data
‎ 
## Team 21, Zero Flux Given

Convolution neural newtork used to classify images of plastic and images of jellyfish for the theme of 'Tech for Good'.
The labelled datasets came from [Jellyfish Object Detection](https://github.com/srv/jf_object_detection) and [DeepData](https://drive.google.com/drive/folders/1fsS_u2QpbRGynYkP6-D6cfvq8r0hpjXI). 

Our motivations are:

- Plastic cleanup to combat pollution
- Jellyfish are an invasive species, their monitoring can help save ecosystems to aid in conservation
- Oceanic survey validation to aid in marine research

## Preprocessing 

In the `preprocessing_jellyfish` and `preprocessing_plastic` folders we have notebooks where we cropped each image to the bounding box, added padding to make all images square, then rescaled so that each image is a 100x100 pixels. 

## Machine learning

We used a convolution neural network with the following layers 

```
model = Sequential([
    # data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, img_depth)), #rescale RGB values to (0,1)
    layers.Conv2D(32, 3, padding='same', activation='relu',input_shape=(img_height, img_width, img_depth)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes)
])
```

with 10 and 100 epochs. We reached a maximum validation accuracy of 0.94.

# Contributors

We are final year students at Imperial College London, partaking in the IBM hackathon. All work is our own and datasets are publicly available at the links above.

## Jacob J. J. Edginton
## Yuet Long Lai
## Ganel R. Nallamilli
## Oliver Phillips
## Ivan Shalashilin
## Pieter Van Steenweghen
