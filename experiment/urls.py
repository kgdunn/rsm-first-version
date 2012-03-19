from django.conf.urls.defaults import *
from rsm.experiment import views
from django.conf import settings


urlpatterns = patterns('',
    (r'take-home-final^$', views.sign_in),
    (r'^take-home-final/$', views.sign_in),
    (r'^take-home-final/not-registered$', views.not_registered_student),
    (r'^take-home-final/run-experiment-(.*)/', views.run_experiment),
    (r'^take-home-final/download-csv-(.*)/', views.download_csv),
    (r'^take-home-final/download-pdf-(.*)/', views.download_pdf),
)


if settings.DEBUG:
    # Small problem: cannot show 404 templates /media/....css file, because
    # 404 gets overridden by Django when in debug mode
    urlpatterns += patterns(
        '',
        (r'^media/(?P<path>.*)$',
         'django.views.static.serve', {'document_root': settings.MEDIA_ROOT}),
    )
