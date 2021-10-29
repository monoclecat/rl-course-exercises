This repository was developed on a Ubuntu 20 machine. 

## DQN Atari
For DQN Atari, make sure the following packages are installed.
```bash
sudo apt --no-install-recommends xvfb python3-opengl ffmpeg
```

Next, download and unpack the Atari ROMs from 
`http://www.atarimania.com/roms/Roms.rar` into `roms/`. 

For some odd reason, unpacking the RAR with the Ubuntu archive manager was akin 
to opening a [zip bomb](https://en.wikipedia.org/wiki/Zip_bomb). 
Instead, try the terminal command `unrar` as shown below. 

Then, import them with 
```bash
sudo apt install unrar
unrar x Roms.rar
# Move the resulting zips into roms/
python -m atari_py.import_roms roms
```
