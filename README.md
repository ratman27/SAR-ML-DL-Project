# SAR Image Colorization using Deep Learning Model 

Synthetic Aperture Radar (SAR) is a type of radar technology used to create detailed images of landscapes and objects, even in bad weather or at night. SAR image is an active microwave-based imaging. Being the wavelength longer, images are not affected by clouds, haze, and other meteorological conditions that otherwise affect visible images.

## PROBLEM: 
Lack of color information and severe speckle noise make image interpretation a challenging task even for well-trained remote sensing experts 

## Idea: 
Take an SAR image and an MS image of the same area; use the colors from the MS(Multispectral) image to colorize the SAR image.

## Our Solution:

### Protocol for creating artificial color SAR images using the SAR-MS image fusion technique:
- We make “artificial” colorized SAR images by combining SAR and MS images
- We train a neural network using artificial images
- We now have a neural network that can colorize SAR images

### **How will we do this?:**
 RGB (red green blue) to IHS (intensity, hue, saturation) conversion:
 We take SAR (black-and-white) and MS (color) images and get their RGB values. Then, we convert the RGB values to IHS values (tools that do this are already available).
 We take the Intensity of the SAR image and the hue and saturation of the MS image (spatial structure information is mainly contained in the intensity (I) component, while the spectral information is contained in the hue 
 (H) and saturation (S) components).
 This allows us to colorize the image without losing out on its spatial features.

## Step by Step process of our project
- We collect SAR data and MS data from https://bhoonidhi.nrsc.gov.in/bhoonidhi/index.html (ISRO’s website where they publish data of all their satellites)
- SAR data and MS data are taken from different satellites (SAR from RISAT - 1, MS from ResourceSat 1,2,3), and take the images they produced over a specific time period (we’re taking June-July 2024)
- We fuse the two images together and once we do this, we have a bunch of “artificial” colorized SAR images.
- We can then use these colorized SAR images to fix our problem of not having enough colorized SAR images to give to our neural network.
- The neural network goes through all the thousands (preferably more) of black and white SAR images and their colorized versions, it then learns how to colorize an SAR image.
- We can now use the neural network to give us a colorized image of any black and white SAR image.




