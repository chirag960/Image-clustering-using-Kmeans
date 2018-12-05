import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

class KMeans :

	def __init__(self, k, threshold , T):
		self.k = k
		self.threshold = threshold
		self.T = T
 
	
	def fit_sample(self, image_array):
		r,d = image_array.shape		

		image_array_sample = shuffle(image_array)[:r/1000]

		#print image_array_sample

		self.centroids = {}
		for i in range(k):
			self.centroids[i] = image_array_sample[i]

		#print centroids

		for i in range(self.T):
			self.classifications = {}
			for i in range(self.k):
				self.classifications[i] = []

			for featureset in image_array_sample:
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				#print classification
				self.classifications[classification].append(featureset)
				'''print("\nfor this loop : ")
				print("distances : ",distances)
				print("classification : ",classification)'''

			#print classifications

			prev_centroids = self.centroids
		
			#calculate the average of the classifications formed and shift the centroids to that average
			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification],axis=0)

			optimized = True
		
			#Check if the shift is less or greater than threshold. IF its less, its towards convergence so stop the iterations
			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.threshold:
					print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
					optimized = False
		
			if optimized:
				print self.centroids
				#print "/n This is classifications : ",self.classifications
				print "It is optimized"
				break

	def get_labels(self,image_array):
		labels = np.zeros((w*h))		
		i=0
		for index,featureset in enumerate(image_array):
			distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
			classification = distances.index(min(distances))
			#print classification			
			#nearest_centroid = self.centroids[classification]
			labels[i] = classification
			i = i+1
			#print nearest_centroid
			#new_image[index] = nearest_centroid
			#print "this is classification: ",classification
		#print labels
		return labels


	def predict_image(self,image_array):		
		d = image_array.shape[1]
		labels = self.get_labels(image_array)
		new_image = np.zeros((w, h, d))
		i = 0
		#for featureset in image_array:
		for j in range(w):
			for k in range(h):
				nearest_centroid = self.centroids[labels[i]]
				#print nearest_centroid	
				i = i+1		
				new_image[j][k] = nearest_centroid
				#print "this is classification: ",classification
		return new_image


'''		
	def get_centroids():
	
class Image_Features():
	
	def __init__():

	def get_total_colors():

'''
#total number of clusters
k = 2

#print full array
np.set_printoptions(threshold=np.nan)

#read the image. Get W*H*3 dimension matrix
image = plt.imread("mini.jpg")
#print(image)
image = np.array(image, dtype=np.float64) / 255
w,h,d =  image.shape

image_array = np.reshape(image, (w * h, d))
#print image_array

#Count number of unique colors in image
u = np.unique(image_array,axis=0)
og_img_total_colors = len(u)
#print "Original Image total Colors = ",og_img_total_colors

clf = KMeans(k, threshold = 0.02, T=500)
clf.fit_sample(image_array)

#labels = clf.get_labels(image_array)

output_file  = 'output.png'
new_image = clf.predict_image(image_array)

plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('K-Means')
#print new_image
plt.imshow(new_image)
plt.show()
plt.imsave(output_file,new_image)
