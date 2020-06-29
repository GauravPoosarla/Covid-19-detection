from django.db import models

# Create your models here.
class Use(models.Model): 
    name = models.CharField(max_length=50) 
    x_ray_img = models.ImageField(upload_to='images/') 