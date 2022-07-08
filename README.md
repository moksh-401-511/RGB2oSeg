# RGB2oSeg
RGB based 2nd order Image Segmentation.<br>
* Segmenting image in top 'n' colors present ['n' is given by user].<br>
* Using RGB image format and KMeans clustering algo.<br>
* Sub-labelling dissconnected pixel sections belonging to same Cluster.
![RGB2oSeg](RGB2oSeg\shown\with\Labelling.jpg)

### Usage
**Note:** Run the following codes in command terminal.<br>
### 1. Clone the repository
```
git clone https://github.com/moksh-401-511/RGB2oSeg.git
```
### 2. Running Segmentation
Paste your image into 'Image' directory present in RGB2oSeg<br>
In command terminal, change to RGB2oSeg directory.
```
# -i : (test_image.png) input image name
# -c : (2/3/4..) number of clusters to segment image
# -s : (1/0) whether to perform sub-segmentation
python RGB-sub-segmentation.py -i micro.png -c 3 -s 1
```
Segmented images will be stored in 'Segmented_images' directory and corresponding labeled matrices (for both segmentation and sub-segmentation) are stored as numpy-binary file in 'labeled_img_matrix' directory.
