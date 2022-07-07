#!/bin/bash
wireshark -k -i WLAN -w $1 -c 5000 -a duration:30