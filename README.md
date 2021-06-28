# pyrometeopy
Tools for experimenting with meteorological data from Bufkit soundings. These are almost entirely 
focused on weather as it relates to wildfire.

## Modules

### formulas

This module contains simple meteorological formulas for things like potential temperature, virtual
temperature, vapor pressure, and specific humidity. There are also formulas for converting between 
units and some meteorological constants.

### bufkit

This module contains code for parsing files in the "Bufkit format." This format is undocumented, or
at least I have not yet been able to find the documentation. I basically reverse engineered the 
file format by reading a lot of the files and seeking out whatever random documentation I could 
find online via searches. [Bufkit][bufkit] is program for viewing sounding data output from 
numerical weather models.

### fire_plumes

This module contains code for analyzing atmospheric stability as it relates to wildfire plumes. 

### fire_environment

This module contains functions for analyzing atmospheric parameters that will influence the fire 
but are not directly related to the fire plume. Thunderstorm out flow wind potential and the 
hot-dry-windy index are examples.

### satellite

This module contains functions for working with GOES data with an emphasis on the fire power data.

[bufkit]: https://training.weather.gov/wdtd/tools/BUFKIT/index.php 
