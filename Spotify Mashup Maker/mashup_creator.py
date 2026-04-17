from tkinter import ttk
import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import os
import re
import webbrowser
import math
import librosa
import soundfile as sf
import pygame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pydub import AudioSegment
import tempfile

# Global variables for feedback, current mashup result, and ML models
current_result = None
feedback_log = []
hit_model = None
outlier_model = None
audio_files = {'original': None, 'mashup': None}
current_mashup_audio = None

# ---------------------------
# 1. Helper: Clean Song Names
# ---------------------------
def clean_song_name(name):
    """
    Remove common remix, live, and featured indicators for better matching.
    Example: "Body On My (feat. Pitbull)" becomes "body on my".
    """
    name = re.sub(r"\(feat\.?.*?\)|\(.*?remix.*?\)|\(live.*?\)|\(.*version.*\)", "", name, flags=re.IGNORECASE)
    return name.strip().lower()

# ---------------------------
# 2. Load Dataset Function
# ---------------------------
def load_dataset(file_path):
    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"Dataset file '{file_path}' not found!")
        return None
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            messagebox.showerror("Error", "Dataset is empty. Please provide a valid dataset.")
            return None

        features = ['tempo', 'loudness', 'acousticness', 'instrumentalness', 'valence']
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            messagebox.showerror("Error", f"Dataset is missing required features: {missing_features}")
            return None

        return df
    except Exception as e:
        messagebox.showerror("Error", f"Error loading dataset: {e}")
        return None

# ---------------------------
# 3. Train Hit Prediction Model (Random Forest)
# ---------------------------
def train_hit_model_rf(df, threshold=60):
    """
    Train a Random Forest classifier to predict whether a song is a "hit".
    We define a song as a hit if track_popularity >= threshold.
    """
    df = df.dropna(subset=['track_popularity', 'tempo', 'loudness', 'acousticness', 'instrumentalness', 'valence'])
    df['hit'] = (df['track_popularity'] >= threshold).astype(int)
    features = ['tempo', 'loudness', 'acousticness', 'instrumentalness', 'valence']
    X = df[features].values
    y = df['hit'].values
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ---------------------------
# 4. Train Outlier Detection Model
# ---------------------------
def train_outlier_model(df, contamination=0.1):
    """
    Train an Isolation Forest to detect outlier songs based on audio features.
    Adjust 'contamination' (default=0.1) to set the expected fraction of outliers.
    """
    features = ['tempo', 'loudness', 'acousticness', 'instrumentalness', 'valence']
    df_filtered = df.dropna(subset=features)
    X = df_filtered[features].values
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(X)
    return clf

# ---------------------------
# 5. Mashup Song Finder Function
# ---------------------------
def find_mashup_song(song_name, df):
    features = ['tempo', 'loudness', 'acousticness', 'instrumentalness', 'valence']
    
    df_filtered = df.dropna(subset=features + ['track_name', 'track_artist']).copy()
    df_filtered['clean_track_name'] = df_filtered['track_name'].apply(lambda x: clean_song_name(str(x)))
    df_filtered = df_filtered.drop_duplicates(
    subset=['clean_track_name', 'track_artist']
)

    
    song_name_clean = clean_song_name(song_name)
    song = df_filtered[df_filtered['clean_track_name'] == song_name_clean]
    
    if song.empty:
        return "Woops! Mashup not possible! Try another song."
    
    song_features = pd.DataFrame([song[features].iloc[0]], columns=features)
    
    n_neighbors = min(10, len(df_filtered))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(df_filtered[features])
    distances, indices = nn.kneighbors(song_features)
    mashup_song = None

    for idx in indices[0]:
        candidate_song = df_filtered.iloc[idx]

        # Exclude same song AND same artist
        if (
            candidate_song['clean_track_name'] != song_name_clean
            and candidate_song['track_artist'] != song.iloc[0]['track_artist']
        ):
            mashup_song = candidate_song
            break

    if mashup_song is None:
        return "No suitable mashup song found. Try another track."

    return {
        "Original Song": song.iloc[0]['track_name'],
        "Mashup Suggestion": mashup_song['track_name'],
        "Artist": mashup_song['track_artist'],
        "Tempo": mashup_song['tempo'],
        "Loudness": mashup_song['loudness'],
        "Acousticness": mashup_song['acousticness'],
        "Instrumentalness": mashup_song['instrumentalness'],
        "Valence": mashup_song['valence']
    }

# ---------------------------
# 6. Function to Share Mashup Suggestions
# ---------------------------
def share_mashup():
    mashup_text = result_text.get(1.0, tk.END).strip()
    if not mashup_text or "Woops!" in mashup_text:
        messagebox.showwarning("No Mashup Found", "Find a mashup first before sharing!")
        return
    
    encoded_text = mashup_text.replace("\n", " %0A")
    share_urls = {
        "WhatsApp": f"https://api.whatsapp.com/send?text={encoded_text}",
        "Twitter": f"https://twitter.com/intent/tweet?text={encoded_text}",
        "Instagram": "https://www.instagram.com/"
    }
    
    def open_share_url(platform):
        webbrowser.open(share_urls[platform])
    
    share_window = tk.Toplevel(root)
    share_window.title("Share Mashup")
    share_window.geometry("300x150")
    share_window.configure(bg="#121212")
    
    tk.Label(share_window, text="Share your mashup suggestion on:", font=("Helvetica", 12), bg="#121212", fg="white").pack(pady=10)
    for platform in share_urls:
        tk.Button(share_window, text=f"📤 Share on {platform}", command=lambda p=platform: open_share_url(p),
                  font=("Helvetica", 12), bg="#1DB954", fg="black", padx=5, pady=5).pack(pady=5)

# ---------------------------
# 7. NEW: Audio Processing Functions
# ---------------------------
def load_audio_file(song_type):
    """Load an audio file (MP3 or WAV) for original or mashup song"""
    global audio_files
    
    filetypes = [("Audio files", "*.mp3 *.wav")]
    file_path = filedialog.askopenfilename(title=f"Select {song_type.title()} Song", filetypes=filetypes)
    
    if not file_path:
        return False  # User cancelled
    
    try:
        # Store file path
        audio_files[song_type] = file_path
        
        # Display filename in the GUI
        if song_type == 'original':
            original_label.config(text=os.path.basename(file_path))
        else:
            mashup_label.config(text=os.path.basename(file_path))
        
        # Analyze and display waveform
        display_waveform(file_path, song_type)
        
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Could not load audio file: {e}")
        return False

def display_waveform(file_path, song_type):
    """Load and display audio waveform"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, duration=60)  # Limit to 60 seconds for display
        
        # Get the appropriate canvas based on song type
        canvas = original_canvas if song_type == 'original' else mashup_canvas
        figure = Figure(figsize=(5, 2), dpi=100)
        figure.patch.set_facecolor('#282828')
        
        plot = figure.add_subplot(111)
        plot.plot(y, color='#1DB954')
        plot.set_facecolor('#282828')
        plot.set_title(f"{song_type.title()} Song Waveform", color='white')
        plot.set_yticks([])
        plot.set_xticks([])
        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.spines['bottom'].set_visible(False)
        plot.spines['left'].set_visible(False)
        
        # Clear previous canvas if it exists
        for widget in canvas.winfo_children():
            widget.destroy()
            
        figure_canvas = FigureCanvasTkAgg(figure, canvas)
        figure_canvas.draw()
        figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Could not display waveform: {e}")

def create_mashup():
    """Create a mashup from the two loaded audio files"""
    global audio_files, current_mashup_audio
    
    if not audio_files['original'] or not audio_files['mashup']:
        messagebox.showwarning("Missing Files", "Please load both original and mashup songs first.")
        return
    
    try:
        # Load audio files using pydub
        original = AudioSegment.from_file(audio_files['original'])
        mashup_track = AudioSegment.from_file(audio_files['mashup'])
        
        # Get tempo information using librosa
        y_orig, sr_orig = librosa.load(audio_files['original'], sr=None)
        y_mash, sr_mash = librosa.load(audio_files['mashup'], sr=None)
        
        tempo_orig, _ = librosa.beat.beat_track(y=y_orig, sr=sr_orig)
        tempo_mash, _ = librosa.beat.beat_track(y=y_mash, sr=sr_mash)
        
        # Adjust tempo if needed (basic implementation)
        if abs(tempo_orig - tempo_mash) > 5:  # If tempos are significantly different
            # Stretch or compress the mashup track to match original tempo
            tempo_ratio = tempo_orig / tempo_mash
            # This is a simple implementation; more advanced time stretching would be better
            mashup_track = mashup_track._spawn(mashup_track.raw_data, overrides={
                "frame_rate": int(mashup_track.frame_rate * tempo_ratio)
            })
            mashup_track = mashup_track.set_frame_rate(original.frame_rate)
        
        # Make sure tracks are same length for simple overlay (take shorter length)
        min_length = min(len(original), len(mashup_track))
        original = original[:min_length]
        mashup_track = mashup_track[:min_length]
        
        # Mix the tracks (50% each)
        mashup_result = original.overlay(mashup_track, gain_during_overlay=-6)
        
        # Create temp file for the mashup
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        mashup_result.export(temp_file.name, format="wav")
        
        current_mashup_audio = temp_file.name
        
        # Display result waveform
        display_mashup_waveform(temp_file.name)
        
        status_label.config(text="Mashup created successfully! Click 'Play Mashup' to listen.")
        return True
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create mashup: {e}")
        return False

def display_mashup_waveform(file_path):
    """Display the created mashup waveform"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Get the canvas
        canvas = result_canvas
        figure = Figure(figsize=(5, 2), dpi=100)
        figure.patch.set_facecolor('#282828')
        
        plot = figure.add_subplot(111)
        plot.plot(y, color='#e74c3c')  # Different color for mashup result
        plot.set_facecolor('#282828')
        plot.set_title("Mashup Result Waveform", color='white')
        plot.set_yticks([])
        plot.set_xticks([])
        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.spines['bottom'].set_visible(False)
        plot.spines['left'].set_visible(False)
        
        # Clear previous canvas if it exists
        for widget in canvas.winfo_children():
            widget.destroy()
            
        figure_canvas = FigureCanvasTkAgg(figure, canvas)
        figure_canvas.draw()
        figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    except Exception as e:
        messagebox.showerror("Error", f"Could not display mashup waveform: {e}")

def init_audio_player():
    """Initialize pygame mixer for audio playback"""
    try:
        pygame.mixer.init()
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Could not initialize audio player: {e}")
        return False

def play_audio(audio_type):
    """Play the selected audio (original, mashup track, or mashup result)"""
    global audio_files, current_mashup_audio
    
    # Stop any currently playing audio
    pygame.mixer.music.stop()
    
    if audio_type == 'original' and audio_files['original']:
        file_to_play = audio_files['original']
    elif audio_type == 'mashup' and audio_files['mashup']:
        file_to_play = audio_files['mashup']
    elif audio_type == 'result' and current_mashup_audio:
        file_to_play = current_mashup_audio
    else:
        messagebox.showwarning("Play Error", f"No {audio_type} audio file loaded.")
        return
    
    try:
        pygame.mixer.music.load(file_to_play)
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showerror("Error", f"Could not play audio: {e}")

def stop_audio():
    """Stop currently playing audio"""
    try:
        pygame.mixer.music.stop()
    except:
        pass

def save_mashup():
    """Save the created mashup to a file"""
    global current_mashup_audio
    
    if not current_mashup_audio:
        messagebox.showwarning("Save Error", "No mashup has been created yet.")
        return
    
    filetypes = [("WAV files", "*.wav"), ("MP3 files", "*.mp3")]
    file_path = filedialog.asksaveasfilename(title="Save Mashup", filetypes=filetypes, defaultextension=".wav")
    
    if not file_path:
        return  # User cancelled
    
    try:
        # Load the temp mashup file
        audio = AudioSegment.from_file(current_mashup_audio)
        
        # Export to the selected format
        file_ext = os.path.splitext(file_path)[1].lower()
        format_type = file_ext.lstrip('.')
        
        audio.export(file_path, format=format_type)
        messagebox.showinfo("Success", f"Mashup saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save mashup: {e}")

# ---------------------------
# 8. Tkinter GUI (Updated)
# ---------------------------
def main():
    global root, result_text, current_result, hit_model, outlier_model
    global original_label, mashup_label, status_label
    global original_canvas, mashup_canvas, result_canvas
    
    file_path = r"C:\Users\Riya\Desktop\hackathon\spotify_songs.csv"

    df = load_dataset(file_path)
    if df is None:
        return

    # Train the hit prediction model and the outlier model
    hit_model = train_hit_model_rf(df, threshold=60)
    outlier_model = train_outlier_model(df, contamination=0.1)

    # Initialize audio player
    init_audio_player()

    root = tk.Tk()
    root.title("🎵 Spotify Mashup Creator")
    root.geometry("900x800")
    root.configure(bg="#121212")
    
    # Title and Song Name Entry
    tk.Label(root, text="🎶 Spotify Mashup Creator 🎶", font=("Helvetica", 20, "bold"), bg="#121212", fg="#1DB954").pack(pady=15)
    
    # Create notebook (tabbed interface)
    notebook = ttk.Notebook(root)
    
    # Tab 1: Song Finder
    finder_frame = tk.Frame(notebook, bg="#121212")
    notebook.add(finder_frame, text="Find Mashup Suggestion")
    
    # Tab 2: Audio Mashup
    audio_frame = tk.Frame(notebook, bg="#121212")
    notebook.add(audio_frame, text="Create Audio Mashup")
    
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ----------------------
    # Tab 1: Song Finder
    # ----------------------
    tk.Label(finder_frame, text="Enter the name of a song:", font=("Helvetica", 14), bg="#121212", fg="white").pack(pady=5)
    
    song_entry = tk.Entry(finder_frame, width=50, font=("Helvetica", 14), bg="#282828", fg="white", relief="flat", borderwidth=2, insertbackground="white")
    song_entry.pack(pady=5)
    
    result_text = scrolledtext.ScrolledText(finder_frame, width=70, height=10, font=("Helvetica", 12), bg="#282828", fg="white", relief="flat", borderwidth=2)
    result_text.pack(pady=10)
    
    # Feedback Frame
    feedback_frame = tk.Frame(finder_frame, bg="#121212")
    feedback_frame.pack(pady=5)
    
    def submit_feedback(feedback):
        global current_result, feedback_log
        if current_result is None:
            messagebox.showinfo("Feedback", "No mashup suggestion to rate yet!")
            return
        feedback_log.append({
            "Original Song": current_result["Original Song"],
            "Mashup Suggestion": current_result["Mashup Suggestion"],
            "Feedback": feedback
        })
        messagebox.showinfo("Feedback", f"Thanks for your feedback: {feedback}")
        print("Feedback Log:", feedback_log)
        current_result = None  # Reset after feedback
    
    tk.Button(feedback_frame, text="👍 Like", font=("Helvetica", 12), bg="#2ecc71", fg="black", padx=10, pady=5,
              command=lambda: submit_feedback("like")).pack(side=tk.LEFT, padx=10)
    tk.Button(feedback_frame, text="👎 Dislike", font=("Helvetica", 12), bg="#f1c40f", fg="black", padx=10, pady=5,
              command=lambda: submit_feedback("dislike")).pack(side=tk.LEFT, padx=10)
    
    # Button Frame for Search and Share
    button_frame = tk.Frame(finder_frame, bg="#121212")
    button_frame.pack(pady=10)
    
    def search_song():
        global current_result
        song_name = song_entry.get().strip()
        if not song_name:
            messagebox.showwarning("Input Error", "Please enter a song name.")
            return
        result = find_mashup_song(song_name, df)
        result_text.delete(1.0, tk.END)
        if isinstance(result, dict):
            current_result = result  # Save result for feedback

            # New ML component: Outlier Detection on the input song.
            features_list = ['tempo', 'loudness', 'acousticness', 'instrumentalness', 'valence']
            input_song = df[df['track_name'].apply(lambda x: clean_song_name(str(x))) == clean_song_name(song_name)]
            if not input_song.empty:
                input_features = input_song.iloc[0][features_list].values.reshape(1, -1)
                outlier_pred = outlier_model.predict(input_features)[0]  # 1 for inlier, -1 for outlier
                uniqueness = "Unique (Outlier)" if outlier_pred == -1 else "Typical"
                result["Song Uniqueness"] = uniqueness

            output = "\n".join([f"{key}: {value}" for key, value in result.items()])
        else:
            current_result = None
            output = result
        result_text.insert(tk.END, output)
    
    tk.Button(button_frame, text="🔍 Find Mashup Song", command=search_song,
              font=("Helvetica", 14, "bold"), bg="#1DB954", fg="black", padx=10, pady=5).pack(side=tk.LEFT, padx=10)
    
    tk.Button(button_frame, text="📤 Share Mashup", command=share_mashup,
              font=("Helvetica", 14, "bold"), bg="#1DB954", fg="black", padx=10, pady=5).pack(side=tk.LEFT, padx=10)
    
    # ----------------------
    # Tab 2: Audio Mashup
    # ----------------------
    
    # Song Loading Section
    loading_frame = tk.Frame(audio_frame, bg="#121212")
    loading_frame.pack(pady=10, fill=tk.X)
    
    # Original Song Section
    original_frame = tk.Frame(loading_frame, bg="#121212", bd=2, relief=tk.GROOVE)
    original_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    tk.Label(original_frame, text="Original Song", font=("Helvetica", 14, "bold"), bg="#121212", fg="#1DB954").pack(pady=5)
    
    original_label = tk.Label(original_frame, text="No file selected", font=("Helvetica", 12), bg="#121212", fg="white")
    original_label.pack(pady=5)
    
    tk.Button(original_frame, text="Load Original Song", command=lambda: load_audio_file('original'),
              font=("Helvetica", 12), bg="#1DB954", fg="black", padx=5, pady=5).pack(pady=5)
    
    tk.Button(original_frame, text="▶️ Play", command=lambda: play_audio('original'),
              font=("Helvetica", 12), bg="#3498db", fg="black", padx=5, pady=5).pack(pady=5)
    
    # Waveform canvas for original song
    original_canvas = tk.Frame(original_frame, bg="#282828", height=150)
    original_canvas.pack(pady=5, fill=tk.BOTH, expand=True)
    
    # Mashup Song Section
    mashup_frame = tk.Frame(loading_frame, bg="#121212", bd=2, relief=tk.GROOVE)
    mashup_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    tk.Label(mashup_frame, text="Mashup Song", font=("Helvetica", 14, "bold"), bg="#121212", fg="#1DB954").pack(pady=5)
    
    mashup_label = tk.Label(mashup_frame, text="No file selected", font=("Helvetica", 12), bg="#121212", fg="white")
    mashup_label.pack(pady=5)
    
    tk.Button(mashup_frame, text="Load Mashup Song", command=lambda: load_audio_file('mashup'),
              font=("Helvetica", 12), bg="#1DB954", fg="black", padx=5, pady=5).pack(pady=5)
    
    tk.Button(mashup_frame, text="▶️ Play", command=lambda: play_audio('mashup'),
              font=("Helvetica", 12), bg="#3498db", fg="black", padx=5, pady=5).pack(pady=5)
    
    # Waveform canvas for mashup song
    mashup_canvas = tk.Frame(mashup_frame, bg="#282828", height=150)
    mashup_canvas.pack(pady=5, fill=tk.BOTH, expand=True)
    
    # Controls Section
    controls_frame = tk.Frame(audio_frame, bg="#121212")
    controls_frame.pack(pady=10, fill=tk.X)
    
    tk.Button(controls_frame, text="🎵 Create Mashup", command=create_mashup,
              font=("Helvetica", 14, "bold"), bg="#e74c3c", fg="white", padx=10, pady=5).pack(pady=10)
    
    status_label = tk.Label(controls_frame, text="Load audio files and click 'Create Mashup'", 
                          font=("Helvetica", 12), bg="#121212", fg="white")
    status_label.pack(pady=5)
    
    # Result Section
    result_frame = tk.Frame(audio_frame, bg="#121212", bd=2, relief=tk.GROOVE)
    result_frame.pack(pady=10, fill=tk.BOTH, expand=True)
    
    tk.Label(result_frame, text="Mashup Result", font=("Helvetica", 14, "bold"), bg="#121212", fg="#e74c3c").pack(pady=5)
    
    # Buttons for result
    result_buttons = tk.Frame(result_frame, bg="#121212")
    result_buttons.pack(pady=5)
    
    tk.Button(result_buttons, text="▶️ Play Mashup", command=lambda: play_audio('result'),
              font=("Helvetica", 12), bg="#3498db", fg="black", padx=5, pady=5).pack(side=tk.LEFT, padx=5)
    
    tk.Button(result_buttons, text="⏹️ Stop", command=stop_audio,
              font=("Helvetica", 12), bg="#95a5a6", fg="black", padx=5, pady=5).pack(side=tk.LEFT, padx=5)
    
    tk.Button(result_buttons, text="💾 Save Mashup", command=save_mashup,
              font=("Helvetica", 12), bg="#2ecc71", fg="black", padx=5, pady=5).pack(side=tk.LEFT, padx=5)
    
    # Waveform canvas for result
    result_canvas = tk.Frame(result_frame, bg="#282828", height=200)
    result_canvas.pack(pady=5, fill=tk.BOTH, expand=True)
    
    root.mainloop()

if __name__ == "__main__":
    main()
    