from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from image2text.views import home_view  # 导入主页视图

urlpatterns = [
    path('admin/', admin.site.urls),
    path('image2text/', include('image2text.urls')),
    path('', home_view, name='home'),  # 设置主页视图为默认路径
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
