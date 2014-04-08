from django.conf.urls.defaults import *
from experiment import views
from django.conf import settings


urlpatterns = patterns('rsm/',
    url(r'^$', views.sign_in),
    url(r'^/$', views.sign_in, name='rsm_sign_in'),
    url(r'^not-registered$', views.not_registered_student),
    url(r'^run-experiment-(.*)/', views.run_experiment),
    url(r'^download-csv-(.*)/', views.download_csv),
    url(r'^download-pdf-(.*)/', views.download_pdf),
)


if settings.DEBUG:
    # Small problem: cannot show 404 templates /media/....css file, because
    # 404 gets overridden by Django when in debug mode
    urlpatterns += patterns(
        '',
        (r'^media-rsm/(?P<path>.*)$',
         'django.views.static.serve', {'document_root': settings.MEDIA_ROOT}),
    )
