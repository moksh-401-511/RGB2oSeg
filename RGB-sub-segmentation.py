import numpy as np
import cv2
import time
import argparse
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import networkx 
from networkx.algorithms.components.connected import connected_components

def show_image(img,name="test image"):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# calculating number of pixels to be reduced according to image size
# using sigmoid function (change constants according to your choice)
def reduction(n):
    ''' n : total pixels in an image
        return:
            0.98 when n>10000
            0    when n<=1084 '''
    sig = (1.34/(1+np.exp(-0.0005*(n-100)))) - 0.36
    
    return sig if sig>0 else 0

# refer https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current
        
def RGB_seg(image, cluster_no, sub_segmentation=True):
    '''RGB Segmentation
    image: numpy array (3D)
    binwidth: bin size for data distribution'''
    
    x, y, z = image.shape                # X(width), Y(height), Z(channels)
    pixel_list = image.reshape((x*y,z))  # 3D (pixel matrix) -> 2D (pixel list with RGB value)
    total_pixels = x*y
    red = reduction(total_pixels)
    
    # taking only 2% of total pixels as no. of pixels is high
    n_pix = total_pixels - int(total_pixels*red)
    index = np.random.choice(total_pixels, n_pix, replace=False)
    reduced_pixels = pixel_list[index]    
    
    # Using K-Means clustering to get centriods and clusters
    kmeans = KMeans(n_clusters = cluster_no, random_state=0, n_init=3).fit(reduced_pixels)
    centriod_list = kmeans.cluster_centers_
    label_list = kmeans.predict(pixel_list)
    
    # segmented image
    seg_img = centriod_list[label_list].reshape((x,y,z)).astype('uint8')
    # segmented labeled image
    labeled_mat = label_list.reshape((x,y)).astype('uint8')
    
    # sub_segmentation
    if sub_segmentation:
        color_seg_img, sub_labeled_img = sub_grouping(labeled_mat, cluster_no)
        return seg_img, labeled_mat, color_seg_img.astype('uint8'), sub_labeled_img
    
    else:
        return seg_img, labeled_mat, None, None

# sub-segmenting image
def sub_grouping(labeled_mat, maximas):
    
    label = np.array([0]*maximas).astype('uint16')

    mat = labeled_mat*100000
    mat = mat.astype('uint64')
    x,y = labeled_mat.shape
    
    # Labelling format: =========================================
    # parent_class*100000 + sub_class_label
    
    #row-wise matrix labeling
    for i in range(x):
        label += 1

        for j in range(y):
            
            curr_pixel = labeled_mat[i,j]
            assign_label = label[curr_pixel]
            mat[i,j] += assign_label

            if j != 0:
                prev_pixel = labeled_mat[i,j-1]
                if curr_pixel != prev_pixel:
                    label[prev_pixel] += 1
    
    
    sublabeled_img = np.zeros((x,y)).astype('uint64')
    colored_grouped_img = np.zeros((x,y,3)).astype('uint16')
    conflict_mat = np.zeros((x-1,y)).astype('uint8')
    conflict_set = set()

    for i in range(1,x):
        conflict_mat[i-1] = labeled_mat[i-1] - labeled_mat[i]

    row_idx, col_idx = np.where(conflict_mat==0)

    for row,col in zip(row_idx,col_idx):
        up_label = mat[row,col]
        curr_label = mat[row+1,col]
        conflict_set.add((up_label,curr_label))
    
    G = to_graph(conflict_set)
    merged_confl_list = list(connected_components(G))
    
    size = len(merged_confl_list)
    red_cls_label = np.random.randint(0,256,size)
    blue_cls_label = np.random.randint(0,256,size)
    green_cls_label = np.random.randint(0,256,size)
    unique_cls_color = list(zip(red_cls_label,green_cls_label,blue_cls_label))
    
    for i,conf_list in enumerate(merged_confl_list):
        mask = np.isin(mat, list(conf_list))
        sublabeled_img[mask] = next(iter(conf_list))
        colored_grouped_img[mask] = unique_cls_color[i] 

    return colored_grouped_img, sublabeled_img

def main():
    
    # parsing arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--Image", type=str, help = "Input Image")
    parser.add_argument("-c", "--Cluster", type=int, help = "No. of clusters")
    parser.add_argument("-s", "--Sub-seg", type=int, help = "Perform sub-segmentation or not")
    args = parser.parse_args()
    
    ip_image = args.Image
    clusters = args.Cluster
    sub_seg = args.Sub_seg
    
    # importing image
    image = cv2.imread("Images/"+ip_image)
    #show_image(image,"input image")
    
    # performing segmentation
    start = time.time()
    seg_img, labeled_img, sub_seg_img, sub_labeled_img = RGB_seg(image, cluster_no = clusters, sub_segmentation=sub_seg)
    print("time taken:", time.time()-start)
    #show_image(sub_labeled_img.astype('uint8'))
    
    # saving segmented image
    cv2.imwrite("Segmented_images/"+ip_image.rpartition('.')[0]+"_seg_img.png", seg_img)
    if sub_seg:
        cv2.imwrite("Segmented_images/"+ip_image.rpartition('.')[0]+"_sub_seg_img.png", sub_seg_img)
    
    # saving segmented labels matrix
    with open('labeled_img_matrix/'+ip_image.rpartition('.')[0]+"_labels.npy", 'wb') as file:
        np.save(file, labeled_img)
        if sub_seg:
            np.save(file, sub_labeled_img)
        
    # to open the labeled_img_matrix file
    #with open('labeled_img_matrix/'+ip_image.rpartition('.')[0]+"_labels.npy", 'rb') as file:
    #    labeled_img = np.load(file)
    #    if sub_seg:
    #        sub_labeled_img = np.load(file)
    
if __name__ == '__main__':
    main()
    
