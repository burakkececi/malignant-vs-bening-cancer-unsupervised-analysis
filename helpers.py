from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from skimage.filters import rank, threshold_otsu
from skimage.morphology import disk
from skimage.io import imread
import pandas as pd

# converting the image to binary to make grayscale as 0 or 1
def thresholding(image):
    return image > threshold_otsu(image)


# removing noise
def mean_filter(image, disk_radius):
    return rank.mean_percentile(image, disk(disk_radius))

# reading images
def load_images(image_paths):
    images = [imread(img_path) for img_path in image_paths]
    return images
    
# elbow method for guessing k number
def elbow(data, cluster_size=(2, 20)):
    kmeans = KMeans(init='k-means++',n_init=10, max_iter=100, random_state=0)
    elbow = KElbowVisualizer(kmeans, k=cluster_size)
    elbow.fit(data)
    elbow.show()
    return elbow
    
# create dataframe with blob images and their classes
def create_dataframe(data1, target1, data2, target2):
    # Create DataFrames
    df1 = pd.DataFrame({"Data": data1, "Target": target1})
    df2 = pd.DataFrame({"Data": data2, "Target": target2})

    # Concatenate DataFrames
    result_df = pd.concat([df1, df2], ignore_index=True)

    return result_df
    
# gives confusion_matrix and classification_report
def report(y_cancer, labels_km):
    # y_cancer is a ground truth labels and 'labels_xx' are the cluster labels
    conf_matrix = confusion_matrix(y_cancer, labels_km)
    classification_rep = classification_report(y_cancer, labels_km)
    
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_rep)