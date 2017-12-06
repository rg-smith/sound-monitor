from sys import byteorder
from array import array
from struct import pack
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import datetime
from scipy.io import wavfile
import os

import pyaudio
import wave
import smtplib
import time
import numpy as np
import pandas as pd
import cPickle

# User inputs:
sender_email='XXXX@XXXX.com' # email account to send messages from
recipient_emails=['XXXX@XXXX.com','XXXX@XXXX.com']
recipient_phone_numbers=[] # optional--phone numbers to email sms to (see: https://en.wikipedia.org/wiki/SMS_gateway)
google_key='' # password. If you use two-step sign-in, you may need an app password: https://support.google.com/accounts/answer/185833
test_email=5 # how often (minutes) to send an email testing that the server is still working
max_duration=50 # how long to run this program (minutes)
THRESHOLD = 3500 # at what amplitude to start recording audio (everything below is considered background noise and ignored)
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100 # rate of audio recording

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
with open('baby_monitor.pkl', 'rb') as fid:
    clf,pca=cPickle.load(fid)

def send_mail(send_from, send_to, subject, text, files=None):
    server = smtplib.SMTP( "smtp.gmail.com", 587 )
    server.starttls()
    server.login( sender_email, google_key )
    assert isinstance(send_to, list)
    msg = MIMEMultipart(
        From=send_from,
        To=COMMASPACE.join(send_to),
        Date=formatdate(localtime=True),
        Subject=subject
    )
    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            msg.attach(MIMEApplication(
                fil.read(),
                Content_Disposition='attachment; filename="%s"' % basename(f),
                Name=basename(f)
            ))
    server.sendmail(send_from, send_to, msg.as_string())
    server.close()


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in xrange(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in xrange(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    rec_length = 0
    snd_started = False

    r = array('h')
    starttime=time.time()
    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True
        elif not silent and snd_started:
            rec_length+=1

        if snd_started and num_silent > 30:
            saveval=1
            break
        if snd_started and rec_length > 100:
            saveval=1
            break
        if time.time()-starttime>(60*5):
            saveval=0
            break
        

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r, saveval

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data, saveval = record()
    if saveval==1:
        data = pack('<' + ('h'*len(data)), *data)
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()
    return saveval

def categorize(fname,pca,rf):
    data = wavfile.read(fname)
    slen=data[0]*2
    if len(data[1])>slen:
        sound = data[1][0:slen]
    else:
        padsize = (slen-len(data[1]))
        psu = int(round(padsize/2))
        psl = int(padsize/2)
        sound = np.lib.pad(data[1],(psl,psu),'constant')
    freq = np.linspace(-data[0]/2,data[0]/2,slen)
    sounds_fft = np.fft.fftshift(np.fft.fft(sound))
    sounds_fft = sounds_fft[abs(freq)<10000]    
    freq = freq[abs(freq)<10000]
    freq_filt=(abs(freq)<2000)&((freq)>0)
    sounds_fft2 = np.abs(sounds_fft[freq_filt])
    smoothed=np.reshape(pd.rolling_mean(sounds_fft2,5),[1,4000])
    filt=np.isnan(smoothed)
    smoothed[filt]=0
    pca_dat = pca.transform(smoothed)
#    print(pca_dat[0,0:5])
    pred = rf.predict(pca_dat)
#    print(pred)
    names=np.array(['DeepVoice','NoVoice','HighVoice','MediumVoice'])
    name=names[pred]
    prob=rf.predict_proba(pca_dat)
    return name,pca_dat,smoothed, prob

count = 0
crycount = 0
time_min = datetime.datetime.now().time().minute
server = smtplib.SMTP( "smtp.gmail.com", 587 )
server.starttls()
server.login(sender_email, google_key)
while count<round(max_duration/test_email):
    starttime1=time.time()
    while time.time()-starttime1<(60*test_email):
    
        if __name__ == '__main__':

            fname='sound'+str(crycount)+'.wav'
            saveval = record_to_file(fname)
            if saveval==1:
                name,pca_dat,smoothed,prob=categorize(fname,pca,clf)
                probhum=prob[0][0]+np.sum(prob[0][2:4])
        #        print("done - result written to demo.wav")
                # Connect to gmail
                server = smtplib.SMTP( "smtp.gmail.com", 587 )
                server.starttls()
                server.login( sender_email, google_key )
                string_to_send= str(crycount)+': Sound detected! ' + str(round(100*probhum)) + '% chance it was from a human' 
                print(string_to_send)
                if probhum>0.01:
                    send_mail(sender_email, recipient_emails, 'Sound detected', string_to_send,
                              files=[fname])
                    server.sendmail(sender_email, recipient_phone_numbers, string_to_send )
                    crycount = crycount+1
    count = count+1
    server.sendmail(sender_email, recipient_emails, 'System test' )
    server.sendmail(sender_email, recipient_phone_numbers, 'System test' )
