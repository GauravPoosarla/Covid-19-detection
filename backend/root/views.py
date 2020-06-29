from django.shortcuts import render
from .detect import *
from .forms import *
# Create your views here. 
def home_view(request): 
    
    if request.method == 'POST':
        form = UserForm(request.POST, request.FILES) 
        image=Predict()
        if form.is_valid(): 
            form.save()
            x_ray_img=request.FILES['x_ray_img'] 
            result = image.predict(x_ray_img)

            return render(request,'root/result.html',{'result':result})
            # return redirect('result') 
    else: 
        form = UserForm() 
        return render(request, 'root/index.html', {'form' : form}) 
  
  
# def success(request): 
#     result=detect()
#     return HttpResponse('successfully uploaded')

