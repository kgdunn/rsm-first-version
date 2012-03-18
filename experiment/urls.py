from django.conf.urls.defaults import *
from rsm.experiment import views

urlpatterns = patterns('',
    (r'take-home-final^$', views.sign_in),
    (r'^take-home-final/$', views.sign_in),
    (r'^take-home-final/not-registered$', views.not_registered_student),
    (r'^take-home-final/run-experiment-(.*)/', views.run_experiment),
    (r'^take-home-final/download-csv-(.*)/', views.download_csv),
    (r'^take-home-final/download-pdf-(.*)/', views.download_pdf),
)

#urlpatterns += patterns('',
    ## For example, files under _images/file.jpg will be retrieved from
    ## settings.MEDIA_ROOT/file.jpg
    #(r'^media/(?P<path>.*)$', 'django.views.static.serve',
     #{'document_root': '/home/kevindunn/django-projects/rsm/media',
      ##views.DJANGO_SETTINGS.MEDIA_ROOT,
      #'show_indexes': False}),
    #)

urlpatterns += patterns('',
# For example, files under _images/file.jpg will be retrieved from
# settings.MEDIA_ROOT/file.jpg
(r'^site_media/(?P<path>.*)$', 'django.views.static.serve',
        {'document_root': views.DJANGO_SETTINGS.MEDIA_ROOT,})
)
