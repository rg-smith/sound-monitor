# sound-monitor
This script uses the computers microphone to monitor sounds. It listens in the background, and starts recording when the sound level rises above a user-defined threshold.
If the threshold is reached, it performs an analysis on the threshold to determine the probability that the sound came from a human voice.
An e-mail (and optionally, text message) is then sent to the user-specified recipients, with an attachment containing the sound clip, and a description of the probability that the sound was from a human voice. 
I have used this script as a simple baby monitor for when my child is napping and I am nearby. It could also be used as a simple security system. However, it requires further testing and I would not recommend using it without some additional method of monitoring or security, until you have thoroughly tested it on your computer and are confident that it works on your system.
