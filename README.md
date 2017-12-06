# sound-monitor
This script uses the computers microphone to monitor sounds. It listens in the background, and starts recording when the sound level rises above a user-defined threshold.
If the threshold is reached, it performs an analysis on the threshold to determine the probability that the sound came from a human voice.
An e-mail (and optionally, text message) is then sent to the user-specified recipients, with an attachment containing the sound clip, and a description of the probability that the sound was from a human voice. 
I have used this script as a simple baby monitor for when my child is napping and I am nearby. It could also be used as a simple security system. However, it requires further testing and I would not recommend using it without some additional method of monitoring or security, until you have thoroughly tested it on your computer and are confident that it works on your system.  

**Method**  
This project was mostly just for fun, but it ended up working pretty well (but far from perfectly!)  
- A training dataset of 600 2 second sound clips was used, some of humans or children talking, some of random background noise. 
- I took the fourier transform of each, then ran principal component analysis to reduce the number of variables. I used the first 80 principal components.
- I then ran a simple random forest model, with the 80 principal components as the predictors. The model learned to predict whether a sound was from a deep voice, mid-range voice, high voice, or background noise.
- The model did ok at distinguishing between these voices, but I grouped them into either 'human' or 'not human' to improve accuracy. For me, when I run it, typically background noises show a ~20% chance of being from a human, and voices from myself or my child have a much higher (~60% or greater) chance of being from a human, as calculated by the random forest model.
- Feel free to share ideas, or add code of your own!
