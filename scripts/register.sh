#!/bin/sh

SERVER=https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com/

curl --data @./register.json -H "Content-Type: application/json" $SERVER/register
