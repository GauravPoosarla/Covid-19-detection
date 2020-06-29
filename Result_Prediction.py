from keras.models import load_model
from keras.preprocessing import image
import numpy as np

classifier = load_model('model_adv.h')

test_image=image.load_img(r'D:/GAURAV/Projects/covid_19_detector/CovidDataset/Val/Covid/4e43e48d52c9e2d4c6c1fb9bc1544f_jumbo.jpeg', target_size = (224, 224))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis = 0)
result=classifier.predict(test_image)
if result >= 0.5:
    prediction='COVID NEGATIVE'
else:
    prediction='COVID POSITIVE'
    
print(prediction)

