# Image_Style_Transfer
Apply artistic styles (for example, Van Gogh) to an input image using deep learning.
### This code is part of a course assignment (Image Processing for Engineers), Which I lectured in 2022. ###

**Code Description:**
This Python script uses OpenCV, PyTorch, and Torchvision to implement Neural Style Transfer, a deep learning technique that blends the style of one image with the content of another. The code loads a pretrained VGG19 model to extract features from the content and style images, computes content and style losses, and iteratively updates a target image to minimize these losses. It also includes helper functions for image preprocessing, feature extraction, and visualization.

**Libraries and Functions:**
1. OpenCV: Used for saving the final stylized image (cv2.imwrite()).
2. PyTorch: Provides the framework for deep learning, including the VGG19 model, tensor operations, and the optimization process.
3. Torchvision: Supplies the pre-trained VGG19 model and utility transformations for image processing.
4. Matplotlib: Used for displaying images during the process.

**Key Functions:**
1. load_image(): Loads and preprocesses input images.
2. get_features(): Extracts content and style features from the images using VGG19.
3. gram_matrix(): Computes the Gram matrix for style representation.
4. style_transfer(): Performs the iterative optimization to achieve style transfer.

**Examples of Real-Life Applications:**
1. Product Design: Style transfer can be used to visualize how artistic patterns or textures would look on physical products like furniture or clothing.
2. Architectural Visualization: Engineers and architects can apply specific textures or artistic styles to building designs for presentation purposes.
3. Game Development: Creates visually unique environments or textures by applying specific art styles to in-game assets, enhancing the aesthetic appeal.
