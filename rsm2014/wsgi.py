"""
WSGI config for rsm2014 project.

This module contains the WSGI application used by Django's development server
and any production WSGI deployments. It should expose a module-level variable
named ``application``. Django's ``runserver`` and ``runfcgi`` commands discover
this application via the ``WSGI_APPLICATION`` setting.

Usually you will have the standard Django WSGI application here, but it also
might make sense to replace the whole Django WSGI application with a custom one
that later delegates to the Django one. For example, you could introduce WSGI
middleware here, or combine a Django application with an application of another
framework.

"""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rsm2014.settings")

import sys
sys.path.append('/var/django/')
sys.path.append('/var/django/rsm/')
sys.path.append('/var/django/rsm/rsm2014/')


#import logging.handlers
#log_file = logging.getLogger('debug')
#log_file.setLevel(logging.DEBUG)
#fh = logging.handlers.RotatingFileHandler('/var/django/rsm/wsgi.log', maxBytes=5000000, backupCount=10)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#fh.setFormatter(formatter)
#log_file.addHandler(fh)
#log_file.debug('Starting WSGI')
#import sys
#log_file.debug('PATH = %s' % str(sys.path))


# This application object is used by any WSGI server configured to use this
# file. This includes Django's development server, if the WSGI_APPLICATION
# setting points here.
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Apply WSGI middleware here.
# from helloworld.wsgi import HelloWorldApplication
# application = HelloWorldApplication(application)
