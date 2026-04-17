# Spotify Mashup Maker

Spotify Mashup Maker is a desktop application that creates intelligent song mashups using machine learning and audio processing techniques. It analyzes song features and generates smooth, playable mashups.

---

## Features

* **Automatic Mashup Generation**
* **ML-based Song Matching**

  * Nearest Neighbors (similarity)
  * Random Forest (hit prediction)
  * Isolation Forest (outlier detection)
* **Audio Processing**

  * librosa (feature extraction)
  * pydub (audio manipulation)
* **Visualization Support** using matplotlib
* **GUI Interface** built with Tkinter
* **Audio Playback** using pygame

---

## Tech Stack

* Python
* Tkinter (GUI)
* NumPy, Pandas
* scikit-learn
* librosa, pydub, soundfile
* matplotlib
* pygame

---

## Project Structure

```bash
spotify-mashup-maker/
│── mashup_creator.py   # Main application (GUI + logic)
│── README.md
│── .gitignore
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/spotify-mashup-maker.git
cd spotify-mashup-maker
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn librosa soundfile pygame matplotlib pydub
```

---

## Usage

```bash
python mashup_creator.py
```

---

## How It Works

1. Load song/audio files
2. Extract audio features using librosa
3. Find similar tracks using Nearest Neighbors
4. Apply ML models for filtering and prediction
5. Combine audio segments into mashups
6. Play or export the final mashup

---

## Key Highlights

* Combines **machine learning + audio engineering**
* Interactive GUI for easy usage
* Supports real-time playback and visualization

---

## Notes

* Ensure audio files are in supported formats (mp3, wav, etc.)
* Some features require proper audio file paths

---

## Contributing

Contributions are welcome! Feel free to improve models or UI.

---

## License

MIT License

---

## Authors

* Dhyeya Aneesh
* Riya Saju Vithayathil
* Prakruti
