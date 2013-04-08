from django.conf.urls.defaults import *
from experiment import views
from django.conf import settings


urlpatterns = patterns('',
    (r'^$', views.sign_in),
    (r'^/$', views.sign_in),
    (r'^not-registered$', views.not_registered_student),
    (r'^run-experiment-(.*)/', views.run_experiment),
    (r'^download-csv-(.*)/', views.download_csv),
    (r'^download-pdf-(.*)/', views.download_pdf),
)


if settings.DEBUG:
    # Small problem: cannot show 404 templates /media/....css file, because
    # 404 gets overridden by Django when in debug mode
    urlpatterns += patterns(
        '',
        (r'^media/(?P<path>.*)$',
         'django.views.static.serve', {'document_root': settings.MEDIA_ROOT}),
    )
