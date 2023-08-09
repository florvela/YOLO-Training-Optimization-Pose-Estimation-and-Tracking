python3 -m venv env &&
source ./env/bin/activate &&
pip3 install -r requirements.txt ;

rm -rf ByteTrack &&
git clone https://github.com/ifzhang/ByteTrack.git &&
cd ByteTrack &&
python3 setup.py -q develop ;
