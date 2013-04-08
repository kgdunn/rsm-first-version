from django.contrib import admin
from experiment.models import Student, Experiment, Token

class TokenAdmin(admin.ModelAdmin):
    list_per_page = 2000
    #list_display = ('student', 'token_string', 'active')

class StudentAdmin(admin.ModelAdmin):
    list_per_page = 200
    list_display = ('first_name', 'student_number', 'email_address', 'runs_used_so_far', 'offset', 'rotation')

class ExptAdmin(admin.ModelAdmin):
    list_per_page = 3000
    list_display = ('student', 'factor_A', 'factor_B', 'factor_C', 'response', 'response_noisy', 'date_time')

admin.site.register(Student, StudentAdmin)
admin.site.register(Experiment, ExptAdmin)
admin.site.register(Token, TokenAdmin)
