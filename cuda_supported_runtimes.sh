#! /bin/bash

OS=rhel7
ARCH=x86_64

BASEDIR=$(dirname $(realpath $0))
AVAILABLE=$(echo $BASEDIR/drivers/$OS/$ARCH/*/ | xargs -n1 basename | sort -V)
LATEST=$(echo "$AVAILABLE" | tail -n1)

VERBOSE=""
DRIVERS=""

function usage() {
cat <<@EOF
Usage: $(basename $0) [-d VERSION] [-h] [-l] [-v]

Options:
  -d VERSION    Try to use the CUDA driver VERSION bundled with this program,
                instead of the version installed on the system.
                Use "latest" to use the most recent driver available.
                Note that a driver newer than the one installed on the system
                can only be used with a datacentre-class GPU.

  -h            Print a short usage and exits.

  -l            Lists the available driver versions bundled with this program.

  -v            Make the underlying test more verbose.
                This is useful for debugging any CUDA device, driver or runtime
                problems, but makes the output unsuitable for automatic parsing.
@EOF
}

while getopts "d:hlv" ARG; do
  case $ARG in
    "d")
      DRIVERS="$OPTARG"
      if [ "$DRIVERS" == "latest" ]; then
        DRIVERS=$LATEST
      fi
      if ! [ -d "$BASEDIR/drivers/$OS/$ARCH/$DRIVERS" ]; then
        echo "$(basename $0): Invalid drivers version '$DRIVERS'"
        echo
        echo "Valid drivers versions are:"
        echo "$AVAILABLE"
        exit 1
      fi
      ;;
    "h")
      usage
      exit 0
      ;;
    "l")
      echo "$AVAILABLE"
      exit 0
      ;;
    "v")
      VERBOSE="-v"
      ;;
    *)
      echo "$(basename $0): Invalid option '$ARG'"
      echo
      usage
      exit 1
      ;;
  esac
done

if [ "$DRIVERS" ]; then
  export LD_LIBRARY_PATH=$(dirname $(realpath $0))/drivers/$DRIVERS:$LD_LIBRARY_PATH
fi

for TEST in $(ls bin/test-*); do
  $TEST $VERBOSE
done
