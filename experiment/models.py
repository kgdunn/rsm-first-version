from django.db import models

class Student(models.Model):
    first_name = models.CharField(max_length=250)
    group_name = models.CharField(max_length=250)
    student_number = models.CharField(max_length=7, unique=True, primary_key=True)
    email_address = models.EmailField(blank=True,)
    runs_used_so_far = models.IntegerField(default=0)
    offset = models.FloatField()
    rotation = models.FloatField()


    def __unicode__(self):
        return u'%s [%s]: %i' % (self.first_name, str(self.student_number), self.runs_used_so_far)

class Experiment(models.Model):

    factor_C_choice = (
            ('H', u'Hexane'),
            ('X', u'Xylene'),
            )

    student = models.ForeignKey(Student)
    factor_A = models.FloatField()
    factor_B = models.FloatField()
    factor_C = models.CharField(choices=factor_C_choice, max_length=10)
    response = models.FloatField()
    response_noisy = models.FloatField()
    date_time = models.DateTimeField(auto_now=False, auto_now_add=True)

    def __unicode__(self):
        return u'%s, (%s, %s, %s) = %s at %s' % (str(self.student), str(self.factor_A), str(self.factor_B), self.factor_C, str(self.response_noisy), str(self.date_time))


class Token(models.Model):
    student = models.ForeignKey(Student)
    token_string = models.CharField(max_length=250)
    active = models.BooleanField(default=True)

    def __unicode__(self):
        return u'%s, %s, %s' % (str(self.student), self.token_string, str(self.active))

