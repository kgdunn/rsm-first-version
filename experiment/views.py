#!/usr/bin/env python
# encoding: utf-8

# Django imports
from django.template import loader, Context
from django.http import HttpResponseRedirect, HttpResponse
from django.conf import settings as DJANGO_SETTINGS
from django.shortcuts import render_to_response
from django.contrib.auth.decorators import login_required
from django.contrib import auth
from django.contrib.auth.models import User

# Built-in imports
import csv
import random
import hashlib
import datetime

# Plotting imports
import matplotlib as matplotlib
from matplotlib.figure import Figure  # for plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ReportLab imports
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (Table, Frame, TableStyle, Image)

# Numpy imports
import numpy as np
from numpy.lib import scimath as SM

# Settings
token_length = 12
max_experiments_allowed = 25
show_result = False

# Command line use
#import sys, os
#sys.path.extend(['/var/django/projects/'])
#os.environ['DJANGO_SETTINGS_MODULE'] = 'rsm.settings'

from rsm.experiment.models import Student, Token, Experiment

# Improvements for next time
#
# * sign in, generate a token, email that token: they have to access all their
#   experiments from that tokenized page: prevents other students look at each
#   others experiments.
# * factors are in a Django, so you can have unlimited number of factors, rename them
#   in one place, set limits, center points (if required), etc

# Experimental conditions
# ===============================
# Baseline and limits
baseline_xA = 98.0
baseline_xB = 35.0
baseline_xC = 'Low'
limits_A = [80, 120]
limits_B = [30, 60]
time_delay = datetime.timedelta(0, 1.5*60*60) # time delay in seconds between experiments

# Start and end point of the linear constraint region
# constraint equation: x+y=2 (in scaled units)
constraint_a = [100, 60]
constraint_b = [120, 48]


# Logging
import logging.handlers
my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler(DJANGO_SETTINGS.LOG_FILENAME,
                                          maxBytes=2000000, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
my_logger.addHandler(fh)
my_logger.debug('A new call to the views.py file')

def get_IP_address(request):
    """
    Returns the visitor's IP address as a string.
    """
    # Catchs the case when the user is on a proxy
    try:
        ip = request.META['HTTP_X_FORWARDED_FOR']
    except KeyError:
        ip = ''
    else:
        # HTTP_X_FORWARDED_FOR is a comma-separated list; take first IP:
        ip = ip.split(',')[0]

    if ip == '' or ip.lower() == 'unkown':
        ip = request.META['REMOTE_ADDR']      # User is not on a proxy
    return ip

def generate_random_token():

    return ''.join([random.choice('ABCEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz2345689') for i in range(token_length)])

def generate_result(the_student, factors, num_runs=0, pure_response=False):
    """
    Generates an experimental result for the student.
    The first error added is always the same, the second error
    added is proportional to the number of runs.
    """
    x1s, x2s, x3s = factors
    if x3s == 'Low':
        x3s = 0.0
    elif x3s == 'High':
        x3s = 1.0

    my_logger.debug('Generating a new experimental result for student number ' + the_student.student_number)

    x1off = (85+120)/2.0    # midpoint
    x1scale = (120-85)/6.0  # a range of 6 units from low to high
    x1 = np.array((np.array(x1s) - x1off)/(x1scale+0.0))

    x2off = (50+30)/2.0     #
    x2scale = (50-30)/6.0
    x2 = np.array((np.array(x2s) - x2off)/(x2scale+0.0))


    # 2012 objective function
    # -----------------------
    # Switch x1 and x2 around: makes more sense in the contect of the 2012
    # problem statement
    x1, x2 = x2, x1
    x2 = -x2
    z_num = 10.2*x1 + 5.6*x2 - 2.9*x1**2 - 3*x2**2 - 12.6*x1*x2
    z_den = (0.1*x1**2 + 0.5*x2**2 + 1)**2


    # 2011 objective function
    # ------------------------
    # Use the scimath library to take powers of negative exponents
    #r = np.sqrt(x1**2 + x2**2 )
    # The "0.05" term can really make it difficult: increases the depth of the trap
    # just next to the optimum
    #phi = 0.05 * np.exp(1 - r + SM.power(x1+1, 0.3) + SM.power(x2+1, 0.2))
    #y = 2 + 9*x1 + 7*x2 - 8*x1**2 - 2*x2**2 - 7*x1*x2
    # y = surface; phi is a modifier;
    # then we add the integer variable interaction (turned on or off by x3s)
    # and the offset
    #offset = 65.0
    # y = (phi * y) + 0.5*x1 + (-0.50*x1**2)*x3s
    # y = np.real(y) * 2.0 + offset

    # Offset of +50c/kg up, but drop off by 2c/kg for using high concentration
    y = np.real(z_num/z_den)*2.0 + the_student.offset + 50.0 - 0.05 *x2s - x3s

    if pure_response:
        y_noisy = y
    else:
        np.random.seed(int(the_student.student_number))
        noise_sd = 0.009 * np.abs(np.max(y)) + 0.001
        y_noisy = y + np.random.normal(loc=0.0, scale=noise_sd) + np.random.normal(loc=0.0, scale=noise_sd, size=num_runs)[-1]
        # Add a slow decay in the response with time:
        y_noisy = y_noisy - num_runs/6.0

    return (y, y_noisy)

def plot_results(expts, the_student):
    """Plots the data into a PNG figure file"""
    factor_A = []
    factor_B = []
    factor_C = []
    response = []
    for entry in expts:
        factor_A.append(entry['factor_A'])
        factor_B.append(entry['factor_B'])
        factor_C.append(entry['factor_C'])
        response.append(entry['response'])

    data_string = str(factor_A) + str(factor_B) + str(factor_C) + str(response)
    filename = hashlib.md5(data_string).hexdigest() + '.png'
    full_filename = DJANGO_SETTINGS.MEDIA_ROOT + filename

    # Offsets for labeling points
    dx = 1.2
    dy = 0.05

    # Create the figure

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    fig = Figure(figsize=(9,7))
    rect = [0.15, 0.1, 0.80, 0.85] # Left, bottom, width, height
    ax = fig.add_axes(rect, frameon=True)
    ax.set_title('Response surface: experiments performed', fontsize=16)
    ax.set_xlabel('Reaction temperature [C]', fontsize=16)
    ax.set_ylabel('Duration [min]', fontsize=16)

    if show_result:
        r = 70         # resolution of surface
        x1 = np.arange(limits_A[0], limits_A[1], step=(limits_A[1] - limits_A[0])/(r+0.0))
        x2 = np.arange(limits_B[0], limits_B[1], step=(limits_B[1] - limits_B[0])/(r+0.0))
        X3_lo = 0.0
        X3_hi = 1.0

        X1, X2 = np.meshgrid(x1, x2)
        Y_lo, Y_lo_noisy = generate_result(the_student, (X1, X2, X3_lo), pure_response=True)
        Y_hi, Y_hi_noisy = generate_result(the_student, (X1, X2, X3_hi), pure_response=True)

        levels_lo = np.linspace(0.0, 100, 25) # np.array(50, 55, 60, 65, 70, 75, 80, 85, 90])
        #levels_hi = np.linspace(1.0, 101, 51)
        CS_lo = ax.contour(X1, X2, Y_lo, colors='#777777', levels=levels_lo, linestyles='solid', linewidths=1)
        #CS_hi = ax.contour(X1, X2, Y_hi, colors='#FF0000', levels=levels_hi, linestyles='dotted', linewidths=1)
        ax.clabel(CS_lo, inline=1, fontsize=10, fmt='%1.0f' )
        #ax.clabel(CS_hi, inline=1, fontsize=10, fmt='%1.0f' )

    # Plot constraint
    ax.plot([constraint_a[0], constraint_b[0]], [constraint_a[1], constraint_b[1]], color="#EA8700", linewidth=2)

    # Baseline marker and label
    ax.text(baseline_xA, baseline_xB, "    Baseline", horizontalalignment='left', verticalalignment='center', color="#0000FF")
    ax.plot(baseline_xA, baseline_xB, 'k.', linewidth=2, ms=20)

    for idx, entry_A in enumerate(factor_A):
        if factor_C[idx] == 'Low':
            ax.plot(entry_A, factor_B[idx], 'k.', ms=20)
        else:
            ax.plot(entry_A, factor_B[idx], 'r.', ms=20)
        ax.text(entry_A+dx, factor_B[idx]+dy, str(idx+1))

    ax.plot(107, 57, 'k.', ms=20)
    ax.text(108, 57, 'Low NaCl concentration', va='center', ha='left')

    ax.plot(107, 59, 'r.', ms=20)
    ax.text(108, 59, 'High NaCl concentration', va='center', ha='left')

    ax.set_xlim(limits_A)
    ax.set_ylim(limits_B)

    # Grid lines
    ax.grid(color='k', linestyle=':', linewidth=1)
    #for grid in ax.yaxis.get_gridlines():
    #     grid.set_visible(False)


    canvas=FigureCanvasAgg(fig)
    my_logger.debug('Saving figure: ' + full_filename)
    fig.savefig(full_filename, dpi=150, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=True)

    return filename

def not_registered_student(request):
    """ Invalid student number received"""
    t = loader.get_template("not_registered_student.html")
    c = Context({})
    return HttpResponse(t.render(c))

def sign_in(request):
    """
    Verifies the user. If they are registered, then proceed with the experimental results
    """
    #my_logger.debug('Sign-in page activated')
    if request.method == 'POST':
        form_student_number = request.POST.get('student_number', '')
        my_logger.debug('Student number (POST: sign_in) = '+ str(form_student_number))

        # Must return an HttpResponseRedirect object by the end of this
        try:
            the_student = Student.objects.get(student_number=form_student_number)
        except Student.DoesNotExist:
            # If student number not in list, tell them they are not registered
            return HttpResponseRedirect('/not-registered')
        else:
            return setup_experiment(request, the_student)

    # Non-POST access of the sign-in page: display the login page to the user
    else:
        my_logger.debug('Non-POST sign-in from %s' % get_IP_address(request))
        return render_to_response('sign_in_form.html')

def get_experiment_list(the_student):
    """ Returns a list of experiments associates with `the_student` (a Django record)"""

    # Create a list of dictionaries: contains their previous experiments
    prev_expts = []
    counter = 1
    for item in Experiment.objects.select_related().filter(student=the_student.student_number):

        ## Time between experiments: 1.5 hours
        #delay_seconds = 90 # 1.5*60*60  # given in seconds
        #now = datetime.datetime.now()
        #delta = datetime.timedelta(0, delay_seconds)

        #if (now - item.date_time) < delta:
            #diff = now+delta
            #prev_expts.append({'factor_A': item.factor_A, 'factor_B': item.factor_B, 'factor_C': item.factor_C,
                        #'response': 'Delayed till %s' % diff.strftime("%d %B, %H:%m") , 'date_time': item.date_time, 'number': counter})
        #else:
        prev_expts.append({'factor_A': item.factor_A, 'factor_B': item.factor_B, 'factor_C': item.factor_C,
                            'response': item.response_noisy, 'date_time': item.date_time, 'number': counter})
        counter += 1
    return prev_expts

def render_next_experiment(the_student):
    """ Setup the dictionary and HTML for the student to enter their next experiment.

    the_student: Django record for the student
    """
    # Get the student's details into the template format
    if the_student.grad_student:
        level = '600'
    else:
        level = '400'
    student = {'name': the_student.first_name + ' ' + the_student.last_name,
               'level': level, 'number': the_student.student_number, 'email': the_student.email_address,
               'runs_used_so_far': the_student.runs_used_so_far}

    prev_expts = get_experiment_list(the_student)

    # Calculate bonus marks
    response = [-10000.0]
    for entry in prev_expts:
        response.append(entry['response'])
    highest_profit = np.max(response)
    #my_logger.debug('Profit = ' + str(highest_profit))
    max_profit = -10
    baseline = 63.5
    student['profit_bonus'] =  3.0 * (highest_profit - baseline) / (max_profit - baseline)
    student['runs_bonus'] = -0.25 * the_student.runs_used_so_far + 5.0

    # Generate a picture of previous experiments
    filename = plot_results(prev_expts, the_student)

    token_string = generate_random_token()
    token_item = Token.objects.get_or_create(token_string=token_string, student=the_student, active=True)

    settings = {'max_experiments_allowed': max_experiments_allowed,
                'token': token_string,
                'figure_filename': DJANGO_SETTINGS.MEDIA_URL + filename}

    my_logger.info('Dealing with student = ' + str(the_student.student_number) + '; has run ' + str(len(prev_expts)) + ' already.')
    t = loader.get_template("deal-with-experiment.html")
    c = Context({'PrevExpts': prev_expts, 'Student': student, 'Settings': settings})
    return HttpResponse(t.render(c))

def setup_experiment(request, the_student):
    """
    Returns the web-page where the student can request a new experiment.
    We can assume the student is already registered.
    """
    my_logger.debug('About to run experiment for student = ' + str(the_student.student_number))
    return render_next_experiment(the_student)

def report_invalid_factors(student_number):
    my_logger.debug('Invalid values for factors received from student ' + student_number)
    t = loader.get_template("invalid-factor-values.html")
    c = Context({})
    return HttpResponse(t.render(c))

def run_experiment(request, token):
    """
    Returns the web-page for the student if the token is valid
    """
    my_logger.debug('Running experiment with token=' + str(token))
    if request.method != 'POST':
        my_logger.debug('Non-POST access to `run_experiment` - student has refreshed page with old token')
        return render_to_response('sign_in_form.html')

    # This is a hidden field
    student_number = request.POST.get('_student_number_', '')
    my_logger.debug('Student number (POST:run_experiment) = '+ str(student_number))
    the_student = Student.objects.get(student_number=student_number)

    # Check if the user had valid numbers
    factor_A = request.POST.get('factor_A', '')
    factor_B = request.POST.get('factor_B', '')
    factor_C = request.POST.get('factor_C', '')
    if factor_C == 'Low':
        pass
    elif factor_C == 'High':
        pass
    else:
        report_invalid_factors(student_number)

    try:
        factor_A, factor_B = np.float(factor_A), np.float(factor_B)
    except ValueError:
        return report_invalid_factors(student_number)
    else:
        my_logger.debug('factor_A = '+ str(factor_A) + '; factor_B = ' + str(factor_B) + '; factor_C = ' + str(factor_C))

    # Check constraints:
    satisfied = True
    if factor_A > 120.0 or factor_A < 80.0:
        satisfied = False
    if factor_B > 60.0 or factor_B < 30.0:
        satisfied = False

    # 2011
    # m = (36.6666666666666667-50.0)/(480.0 - 450.0)
    # c = 50.0 - m * 450.0

    # 2012
    m = (48.0 - 60.0)/(120.0 - 100.0)
    c = 60.0 - m * 100.0
    if factor_A*m + c < factor_B:    # predicted_B > actual_B: then you are in the constraint region
        satisfied = False

    if not satisfied:
        my_logger.debug('Invalid values for factors received from student ' + student_number)
        t = loader.get_template("invalid-factor-values.html")
        c = Context({})
        return HttpResponse(t.render(c))

    # Check if the user has enough experiments remaining
    if the_student.runs_used_so_far >= max_experiments_allowed:
        # Used token
        my_logger.debug('Limit reached for student number ' + student_number)
        t = loader.get_template("experiment-limit-reached.html")
        c = Context({})
        return HttpResponse(t.render(c))

    # Check that the token matches the student number and hasn't been used already
    token_item = Token.objects.filter(token_string=token)
    token_pk = token_item[0].pk
    if not token_item[0].active:
        # Used token
        my_logger.debug('Used token received: ' + token)
        t = loader.get_template("experiment-already-run.html")
        c = Context({})
        return HttpResponse(t.render(c))

    if token_item[0].student != the_student:
        my_logger.debug('Token does not belong to student')
        t = loader.get_template("token-spoofing.html")
        c = Context({})
        return HttpResponse(t.render(c))

    response, response_noisy = generate_result(the_student, [factor_A, factor_B, factor_C], num_runs=the_student.runs_used_so_far+1)

    # Check the time between experiments:
    now = datetime.datetime.now()
    last_run = datetime.datetime(1, 1, 1)
    for item in Experiment.objects.filter(student=the_student.student_number):
        if item.date_time > last_run:
            last_run = item.date_time

    if (now - last_run) < time_delay:
        #time_done = last_run + delta
        my_logger.debug('Experiments too close from student ' + student_number)
        t = loader.get_template("too-close-in-time.html")
        c = Context({})
        return HttpResponse(t.render(c))

    else:
        time_done = now

    expt = Experiment.objects.get_or_create(student=the_student,
                                            factor_A=factor_A,
                                            factor_B=factor_B,
                                            factor_C=factor_C,
                                            response=response,
                                            response_noisy=response_noisy,
                                            date_time=time_done)

    the_student.runs_used_so_far = the_student.runs_used_so_far + 1
    the_student.save()

    token = Token.objects.get(pk=token_pk)
    token.active = False
    token.save()

    return render_next_experiment(the_student)

def download_csv(request, token):
    """ From the download link on the output"""
    token_item = Token.objects.filter(token_string=token)
    the_student = token_item[0].student
    my_logger.debug('Generating CSV file for token = ' + str(token) + '; student number = ' + the_student.student_number)

    prev_expts = get_experiment_list(the_student)

    # Create CSV response object
    response = HttpResponse(mimetype='text/csv')
    response['Content-Disposition'] = 'attachment; filename=takehome-group-' + the_student.student_number + '-' + token + '.csv'
    writer = csv.writer(response)
    writer.writerow(['Number', 'DateTime', 'Reactor temperature [C]', 'Duration [min]', 'NaCl concentration', 'Profit [c/kg]'])
    #writer.writerow(['0','Baseline','93.0','50.0','Low','63.5'])
    for expt in prev_expts:
        writer.writerow([str(expt['number']),
                         expt['date_time'].strftime('%d %B %Y %H:%M:%S'),
                         str(expt['factor_A']),
                         str(expt['factor_B']),
                         str(expt['factor_C']),
                         str(round(expt['response'],1)),
                         ])
    return response

def download_pdf(request, token):
    """ From the download link on the output"""
    token_item = Token.objects.filter(token_string=token)
    the_student = token_item[0].student
    my_logger.debug('Generating PDF file for token = ' + str(token) + '; student number = ' + the_student.student_number)
    PDF_filename = 'takehome-group-%s-%s.pdf' % (the_student.student_number, token)
    response = HttpResponse(mimetype='application/pdf')
    response['Content-Disposition'] = 'attachment; filename=%s' % PDF_filename

    c = canvas.Canvas(response, pagesize=letter)
    W, H = letter
    RMARGIN = LMARGIN = 15*mm
    TMARGIN = 15*mm
    BMARGIN = 15*mm

    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(W/2, H-TMARGIN, '4C3/6C3 take-home exam: question 4 report')
    text = c.beginText(LMARGIN, H-TMARGIN-10*mm)
    text.setFont("Helvetica-Bold", 14)
    text.textLines('Student name(s): %s\n' % the_student.first_name)
    text.textLines('Group number: %s\n' % the_student.student_number)
    c.drawText(text)

    # Collect experimental results together:
    frameWidth = W - (LMARGIN + RMARGIN)
    frameHeight = H - (TMARGIN + BMARGIN+30*mm)
    frame = Frame(LMARGIN, BMARGIN, frameWidth, frameHeight, showBoundary=0)
    table_data = [['Run', 'Date/Time of experiment', 'Reaction temperature [C]', 'Duration [min]', 'NaCl concentration', 'Profit [c/kg]']]

    prev_expts = get_experiment_list(the_student)
    for expt in prev_expts:
        table_data.append([str(expt['number']),
                           expt['date_time'].strftime('%d %B %Y %H:%M:%S'),
                           str(expt['factor_A']),
                           str(expt['factor_B']),
                           str(expt['factor_C']),
                           str(round(expt['response'],1))])

    tblStyle = TableStyle([('BOX',(0,0), (-1,-1), 2,colors.black),
                           ('BOX',(0,0), (-1,0), 1,colors.black),
                           ('FONT',(0,0), (-1,0), 'Helvetica-Bold',10)])
    table_obj = Table(table_data, style=tblStyle)


    frame.addFromList([table_obj], c)

    c.showPage()

    filename = DJANGO_SETTINGS.MEDIA_ROOT + plot_results(prev_expts, the_student)

    # When passing a filename: requires the Python Imaging Library (PIL)
    c.drawImage(filename, LMARGIN, BMARGIN,
                width=0.85*W, preserveAspectRatio=True, anchor='sw')
    c.showPage()
    c.save()

    return response
