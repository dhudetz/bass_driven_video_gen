# bass_driven_video_gen
generate music videos that randomly splice .MOV clips to the bass hits of an audio clip into an MP4 file

## Remaining Items
* Feature: get a proxy for the amount of motion in each frame. Use this to line up high energy clips with high energy parts of the music.
* Feature: toggle using repeat footage. Best simple algo might be to cut the biggest clips first
* User interface -> design bass hit profile then compile
   * stretch goal on UI: thread each compilation to make multiple instances of the video at once
* Save configurations as its own format load different formats
* Other modules to replace the bass detector would be a CV or LLM based decision maker.
  
