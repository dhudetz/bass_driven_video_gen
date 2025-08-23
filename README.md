# bass_driven_video_gen
## Main features:
* Automatically generate an entire music video from a folder of video clips with a single button.
* Detect high energy bass hits in audio clip to be used for automatic transitions.
* Visualize bass detections on a plot that automatically updates.
* Select a frequency range and cutoff to create video transitions at different audio moments.
* Save configuration files to pick up where you left off.
* Shell script for a function which runs the tool from any folder.

## How to setup
1. Install `ffmpeg` package on machine. `brew install ffmpeg`
2. Add line in your shell rc file pointing towards where you saved the run script: `source /Users/dhudetz/code/bass_driven_video_gen/run`

## How to use
Two ways to use this tool:
* Run `musicvideo` in any directory with video files in it. All files in the folder will be included in the music video.
* Run `musicvideo <dir-path>` to point towards a specific directory of video files.
* Select your parameters and click `Generate` to start the video compilation process.
* Completed videos are outputted in the same directory you included your clips.
  
 <img width="1035" height="747" alt="image" src="https://github.com/user-attachments/assets/e96a8ae1-9063-4f95-84b2-1309c1384fc6" />
