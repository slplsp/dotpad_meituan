# image2text/urls.py
from django.urls import path
from . import views

from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.image_to_text_view, name='image_to_text_view'),  # 上传页面 URL
    path('result/', views.result_view, name='result_view'),  # 结果页面 URL
    path('delete_images/', views.delete_images_view, name='delete_images'),  # 删除页面 URL
    path('get-array/', views.get_2d_array, name='get_array'),  # 提供二维数组的 API
]
print(urlpatterns)
