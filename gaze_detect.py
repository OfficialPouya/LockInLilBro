import cv2
import numpy as np
import time
import math
import os
import random
import threading
import subprocess
import platform

# suppress warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

#setting macros
calib_factor = 0.5
dist_threshold = 0.25
distraction_timer = 5
eyes_closed_macro_perc = 0.85

class VideoPlayer:
    """Separate video player that uses system's default video player"""
    def __init__(self, video_folder="videos"):
        self.video_folder = video_folder
        self.is_playing = False
        self.current_process = None

    def play_random_video_opencv(self):
        """Play random video using OpenCV (guaranteed auto-play, but simple window)"""
        if not os.path.exists(self.video_folder):
            print(f"Warning: Video folder '{self.video_folder}' not found!")
            return False
            
        video_files = [f for f in os.listdir(self.video_folder) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'))]
        
        if not video_files:
            print(f"No video files found in '{self.video_folder}'!")
            return False
        
        # select random video
        video_file = random.choice(video_files)
        video_path = os.path.join(self.video_folder, video_file)
        
        print(f"Playing video with OpenCV: {video_file}")
        
        def play_video():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            
            cv2.namedWindow("Distraction Alert - Watch This!", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Distraction Alert - Watch This!", 800, 600)
            
            while self.is_playing:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                cv2.imshow("Distraction Alert - Watch This!", frame)
                
                # Break if user presses ESC or window is closed
                if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
                    break
                if cv2.getWindowProperty("Distraction Alert - Watch This!", cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            cap.release()
            cv2.destroyWindow("Distraction Alert - Watch This!")
    
        # Start playing in a thread
        self.is_playing = True
        threading.Thread(target=play_video, daemon=True).start()
        return True


    def play_random_video(self):
        if not os.path.exists(self.video_folder):
            print(f"Warning: Video folder '{self.video_folder}' not found!")
            return False
            
        video_files = [f for f in os.listdir(self.video_folder) 
                        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'))]
        
        if not video_files:
            print(f"No video files found in '{self.video_folder}'!")
            return False
        
        # select random video
        video_file = random.choice(video_files)
        video_path = os.path.abspath(os.path.join(self.video_folder, video_file))
        
        print(f"Playing video: {video_file}")
        
        def play_video():
            try:
                system = platform.system()
                
                if system == "Darwin":  # macOS
                    # Use AppleScript to open and autoplay in QuickTime with looping
                    applescript = f'''
                    tell application "QuickTime Player"
                        activate
                        open POSIX file "{video_path}"
                        tell front document
                            set looping to true
                            play
                        end tell
                    end tell
                    '''
                    self.current_process = subprocess.Popen(
                        ['osascript', '-e', applescript],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    print(f"Video playing in QuickTime Player with looping enabled")
                    
                elif system == "Windows":
                    # Use default Windows video player
                    self.current_process = subprocess.Popen(['start', '', video_path], shell=True)
                else:  # Linux
                    # Try common Linux video players
                    try:
                        self.current_process = subprocess.Popen(['xdg-open', video_path])
                    except:
                        self.current_process = subprocess.Popen(['vlc', video_path])
                
            except Exception as e:
                print(f"Error opening video: {e}")
                self.is_playing = False
                return
        
        # Start playing in a thread
        self.is_playing = True
        threading.Thread(target=play_video, daemon=True).start()
        return True
    
    def pause_video(self):
            """Pause the video playback"""
            try:
                system = platform.system()
                
                if system == "Darwin":  # macOS
                    applescript = '''
                    tell application "QuickTime Player"
                        tell front document
                            pause
                        end tell
                    end tell
                    '''
                    subprocess.run(['osascript', '-e', applescript], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
                    print("Video paused")
                elif system == "Windows":
                    # Windows handling would go here
                    pass
                    
            except Exception as e:
                print(f"Error pausing video: {e}")

    def resume_video(self):
        """Resume the video playback"""
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                applescript = '''
                tell application "QuickTime Player"
                    tell front document
                        play
                    end tell
                end tell
                '''
                subprocess.run(['osascript', '-e', applescript], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
                print("Video resumed")
            elif system == "Windows":
                # Windows handling would go here
                pass
                
        except Exception as e:
            print(f"Error resuming video: {e}")

    def stop_video(self):
            """Stop the video (close the player window)"""
            if self.is_playing:
                try:
                    self.is_playing = False
                    
                    # Try to terminate the video player process
                    if self.current_process:
                        try:
                            self.current_process.terminate()
                            self.current_process.wait(timeout=1)
                        except:
                            try:
                                self.current_process.kill()
                            except:
                                pass
                        self.current_process = None
                    
                except Exception as e:
                    print(f"Warning: Could not stop video player: {e}")
                finally:
                    self.is_playing = False

                
class DistractionDetector:
    def __init__(self):
        # load face and eye detectors
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # tracking
        self.distraction_start_time = None
        self.is_distracted = False
        self.distraction_count = 0
        self.total_distraction_time = 0
        
        # eye closing tracking
        self.eye_close_start_time = None
        self.eyes_closed_counter = 0
        self.eyes_open_counter = 0
        self.eye_state_buffer = []  # Buffer to track eye state over multiple frames
        self.eye_state_history = []  # Track eye state history
        self.blink_detected = False
        
        # Settings
        self.DISTRACTION_THRESHOLD = distraction_timer # time in seconds
        self.FACE_DISTANCE_THRESHOLD = dist_threshold
        self.MIN_FACE_SIZE = 0.15  # min 15% of frame width
        
        # Eye closure settings
        self.EYE_CLOSURE_FRAMES_THRESHOLD = 10  # Number of consecutive frames with no eyes to consider them closed
        self.EYE_REOPEN_FRAMES_THRESHOLD = 3    # Number of consecutive frames with eyes to consider them open again
        self.BLINK_FRAME_THRESHOLD = 5          # Blink is short closure (1-5 frames)
        self.EYE_CLOSE_TIME_THRESHOLD = 2.0     # Seconds before considering eyes closed as distraction
        
        #calibration
        self.screen_center_x = calib_factor
        self.screen_center_y = calib_factor
        self.is_calibrated = False
        self.calibration_frames = []
        
        # video player
        self.video_player = VideoPlayer("videos")
        self.video_thread = None
        self.extended_distraction_start = None
        self.video_played_this_distraction = False
        self.focus_regained_time = None  
        
    def calibrate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Calculate normalized face position
            face_center_x = (x + w/2) / frame.shape[1]
            face_center_y = (y + h/2) / frame.shape[0]
            
            self.calibration_frames.append((face_center_x, face_center_y, w/frame.shape[1]))
            
            if len(self.calibration_frames) >= 30:  # Collect 30 frames
                avg_x = np.mean([f[0] for f in self.calibration_frames])
                avg_y = np.mean([f[1] for f in self.calibration_frames])
                avg_size = np.mean([f[2] for f in self.calibration_frames])
                
                self.screen_center_x = avg_x
                self.screen_center_y = avg_y
                self.normal_face_size = avg_size
                self.is_calibrated = True
                
                print(f"Calibration complete! Normal position: ({avg_x:.2f}, {avg_y:.2f})")
                return True
        
        return False
    
    #eye detection
    def detect_face_and_eyes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            # eye state buffer
            self.eye_state_buffer.append(False)
            if len(self.eye_state_buffer) > 20:
                self.eye_state_buffer.pop(0)
            return None
        
        # get the user, not the people around
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # detect eyes in the upper half of the face
        roi_gray = gray[y:y + int(h/2), x:x+w]
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(20, 20),
            maxSize=(70, 70)
        )
        
        # filter eyes so biggest ones get picked
        filtered_eyes = []
        if len(eyes) > 0:
            # sort
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)
            for i in range(min(2, len(eyes))):
                if i == 0:
                    filtered_eyes.append(eyes[i])
                else:
                    # check if its the same person
                    ex1, ey1, ew1, eh1 = eyes[0]
                    ex2, ey2, ew2, eh2 = eyes[i]
                    distance = math.sqrt((ex2 - ex1)**2 + (ey2 - ey1)**2)
                    if distance > 20:  # min distance between eyes
                        filtered_eyes.append(eyes[i])
        
        eyes = filtered_eyes
        eye_count = len(eyes)
        
        # update eye state tracking
        eyes_detected = eye_count >= 1
        
        # add to buffer
        self.eye_state_buffer.append(eyes_detected)
        if len(self.eye_state_buffer) > 20:
            self.eye_state_buffer.pop(0)
        
        # determine if eyes are consistently closed
        if len(self.eye_state_buffer) >= 5:
            recent_eye_states = self.eye_state_buffer[-5:]
            eyes_closed_percentage = 1 - (sum(recent_eye_states) / len(recent_eye_states))
            
            if eyes_closed_percentage > eyes_closed_macro_perc:
                eyes_consistently_closed = True
            else:
                eyes_consistently_closed = False
        else:
            eyes_consistently_closed = False
        
        # check for blink 
        self.eye_state_history.append(eyes_detected)
        if len(self.eye_state_history) > 30:
            self.eye_state_history.pop(0)
        
        self.blink_detected = False
        if len(self.eye_state_history) >= 10:
            # blinking pattern
            recent_history = self.eye_state_history[-10:]
            for i in range(len(recent_history) - 2):
                if (recent_history[i] and  # open
                    not recent_history[i+1] and  # closed
                    recent_history[i+2]):  # open again
                    if i+2 - i <= self.BLINK_FRAME_THRESHOLD: 
                        self.blink_detected = True
                        break
        
        # calc metrics
        face_center_x = (x + w/2) / frame.shape[1]
        face_center_y = (y + h/2) / frame.shape[0]
        face_size_ratio = w / frame.shape[1]
        
        return {
            'rect': (x, y, w, h),
            'eyes': eyes,
            'eye_count': eye_count,
            'center': (face_center_x, face_center_y),
            'size_ratio': face_size_ratio,
            'eyes_detected': eyes_detected,
            'eyes_consistently_closed': eyes_consistently_closed,
            'blink_detected': self.blink_detected
        }
    
    #distraction check
    def check_distraction(self, face_data, frame_shape, current_time):
        if face_data is None:
            return True, "No face detected", 0
        
        x, y, w, h = face_data['rect']
        eye_count = face_data['eye_count']
        face_center_x, face_center_y = face_data['center']
        face_size = face_data['size_ratio']
        eyes_consistently_closed = face_data['eyes_consistently_closed']
        blink_detected = face_data['blink_detected']
        
        #distance from calibrated center
        distance = math.sqrt((face_center_x - self.screen_center_x)**2 + (face_center_y - self.screen_center_y)**2)
        
        # multiple distraction conditions
        distractions = []
        
        # face too far from center
        if distance > self.FACE_DISTANCE_THRESHOLD:
            distractions.append(f"Face off-center")
        
        # eyes consistently closed (not just blinking)
        if eyes_consistently_closed and not blink_detected:
            distractions.append(f"Eyes closed")
            
            # start or update eye closure timer
            if self.eye_close_start_time is None:
                self.eye_close_start_time = current_time
            else:
                eye_close_duration = current_time - self.eye_close_start_time
                if eye_close_duration > self.EYE_CLOSE_TIME_THRESHOLD:
                    distractions.append(f"Eyes closed too long")
        else:
            # reset eye close time 
            self.eye_close_start_time = None
        
        # looking down/away (no eyes detected but face detected)
        if eye_count < 1 and not eyes_consistently_closed:
            distractions.append(f"Eyes not visible")
        
        # leaning back
        if face_size < self.MIN_FACE_SIZE:
            distractions.append(f"Face too small")
        
        # too close
        if face_size > 0.5:
            distractions.append(f"Too close to screen")
        
        is_distracted = len(distractions) > 0
        reason = " | ".join(distractions) if distractions else "Focused"
        
        return is_distracted, reason, distance
    
   # distraction timer
    def update_distraction_state(self, is_distracted, current_time):
        if is_distracted:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
                self.distraction_count += 1
                self.is_distracted = True
                self.video_played_this_distraction = False
                self.focus_regained_time = None  # Reset focus timer
            
            distraction_duration = current_time - self.distraction_start_time
            
            # check for extended distraction (30+ seconds) and haven't played video yet
            if distraction_duration > 30 and not self.video_played_this_distraction:
                if self.extended_distraction_start is None:
                    self.extended_distraction_start = current_time
                    print(f"Starting video after 30 seconds of distraction")
                
                # Play video in a separate thread
                if not self.video_player.is_playing:
                    print(f"Attempting to play video...")
                    # Store reference to prevent garbage collection
                    self.video_thread = threading.Thread(
                        target=self.video_player.play_random_video,
                        daemon=True
                    )
                    self.video_thread.start()
                    self.video_played_this_distraction = True
            
            # If video is playing and user is still distracted, make sure it's playing (not paused)
            if self.video_player.is_playing and distraction_duration > 30:
                # Resume video if it was paused
                self.video_player.resume_video()
                self.focus_regained_time = None  # Reset focus timer

            if distraction_duration >= 1.0:
                self.total_distraction_time = max(self.total_distraction_time, distraction_duration)
            
            return distraction_duration
        else:
            # User has regained focus
            if self.video_player.is_playing:
                # Start tracking how long they've been focused
                if self.focus_regained_time is None:
                    self.focus_regained_time = current_time
                    # Pause the video when focus is regained
                    self.video_player.pause_video()
                    print("Focus regained - Video paused")
                
                # Check if they've maintained focus for 5 seconds
                focus_duration = current_time - self.focus_regained_time
                if focus_duration >= 5:
                    # Close the video player after 5 seconds of maintained focus
                    print("Focus maintained for 5 seconds - Closing video")
                    self.video_player.stop_video()
                    self.focus_regained_time = None
            else:
                self.focus_regained_time = None
            
            self.distraction_start_time = None
            self.is_distracted = False
            self.extended_distraction_start = None
            
            return 0

def main():
    print("="*60)
    print("SCREEN DISTRACTION DETECTOR")
    print("Detects when you look away at phone or other screens")
    print("="*60)
    print("\nInstructions:")
    print("1. Sit in your normal working position")
    print("2. Look at the screen center during calibration")
    print("3. System will alert when you look away")
    print("4. Video will play if distracted for 30+ seconds")
    print("5. Video opens in system's default media player")
    print("6. Press 'q' to quit, 'r' to reset, 'c' to recalibrate")
    print("="*60)
    
# init
    detector = DistractionDetector()
    cap = cv2.VideoCapture(0)
    
    # Give camera time to initialize
    time.sleep(2)
    
    if not cap.isOpened():
        print("Error: Could not open camera. Trying alternative method...")
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        time.sleep(2)
    
    # camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # statistics
    session_start = time.time()
    last_alarm_time = 0
    alarm_interval = 0.5  # seconds between alarms
    
    # calibration phase
    print("\nStarting calibration... Look at the screen center")
    calibration_complete = False
    calibration_frame_count = 0
    
    while not calibration_complete:
        ret, frame = cap.read()
        if not ret:
            print("Camera Failed")
            return
        
        frame = cv2.flip(frame, 1)
        
        # calibration instructions
        cv2.putText(frame, "CALIBRATION: Look at screen center", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frames collected: {calibration_frame_count}/30", 
                   (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw target
        h, w = frame.shape[:2]
        cv2.circle(frame, (w//2, h//2), 10, (0, 255, 255), -1)
        cv2.circle(frame, (w//2, h//2), 15, (0, 255, 255), 2)
        
        # calibration attempt
        if detector.calibrate(frame):
            calibration_frame_count = len(detector.calibration_frames)
            
        if detector.is_calibrated:
            calibration_complete = True
            break
        
        cv2.imshow("Distraction Detector - Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if detector.video_player:
                detector.video_player.stop_video()
                time.sleep(0.1)
            break
    
    print("\nCalibration complete!")
    
    # main detection loop
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        current_time = time.time()
        frame_count += 1
        
        face_data = detector.detect_face_and_eyes(frame)
        is_distracted, reason, distance = detector.check_distraction(face_data, frame.shape, current_time)
        distraction_duration = detector.update_distraction_state(is_distracted, current_time)
        h, w = frame.shape[:2]
        screen_x = int(detector.screen_center_x * w)
        screen_y = int(detector.screen_center_y * h)
        cv2.circle(frame, (screen_x, screen_y), 8, (255, 255, 0), -1)
        cv2.circle(frame, (screen_x, screen_y), 12, (255, 255, 0), 2)
        
        # Draw face and connection line if face detected
        if face_data is not None:
            x, y, face_w, face_h = face_data['rect']
            eye_count = face_data['eye_count']
            eyes_consistently_closed = face_data['eyes_consistently_closed']
            blink_detected = face_data['blink_detected']
            
            # Draw face rectangle
            color = (0, 255, 0) if not is_distracted else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), color, 2)
            
            # Draw face center and line to screen center
            face_center_x = x + face_w // 2
            face_center_y = y + face_h // 2
            cv2.circle(frame, (face_center_x, face_center_y), 4, color, -1)
            cv2.line(frame, (face_center_x, face_center_y), 
                    (screen_x, screen_y), color, 2)
            
            #eye status indicator
            eye_status_color = (0, 255, 0)  # green for open
            eye_status_text = "Eyes: Open"
            
            if eyes_consistently_closed:
                if blink_detected:
                    eye_status_color = (0, 255, 255)  # yellow for blink
                    eye_status_text = "Eyes: Blink"
                else:
                    eye_status_color = (0, 0, 255)  # red for closed
                    eye_status_text = "Eyes: Closed"
            elif eye_count < 1:
                eye_status_color = (255, 165, 0)  # orange for not detected
                eye_status_text = "Eyes: Not visible"
            
            cv2.putText(frame, eye_status_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_status_color, 2)
            
            # draw eyes if detected
            if eye_count > 0:
                roi_color = frame[y:y + int(face_h/2), x:x+face_w]
                eyes = face_data['eyes']
                for (ex, ey, ew, eh) in eyes:
                    eye_color = (0, 0, 255) if eyes_consistently_closed else (0, 255, 255)
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), eye_color, 2)
        
        # Display status
        status_color = (0, 255, 0) if not is_distracted else (0, 0, 255)
        status_text = "FOCUSED" if not is_distracted else "DISTRACTED"
        
        cv2.putText(frame, f"Status: {status_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Display reason if distracted
        if is_distracted:
            cv2.putText(frame, f"Reason: {reason}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # Show distraction timer
            if distraction_duration > 0:
                timer_text = f"Time: {distraction_duration:.1f}s"
                cv2.putText(frame, timer_text, 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # show video warning if approaching 30 seconds
                if distraction_duration > 25 and distraction_duration <= 30:
                    warning_text = f"Video will play in {30-int(distraction_duration)}s if you don't focus!"
                    text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    text_x = (w - text_size[0]) // 2
                    cv2.putText(frame, warning_text, (text_x, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        # alarm for extended distraction
        if is_distracted and distraction_duration > detector.DISTRACTION_THRESHOLD:
            # red border as alarm
            if current_time - last_alarm_time > alarm_interval:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
                last_alarm_time = current_time
            
            # warning message
            warning_text = f"ON YOUR PHONE! Look back at screen!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, warning_text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # stats
        session_duration = current_time - session_start
        if session_duration > 0:
            focus_time = session_duration - detector.total_distraction_time
            focus_percent = (focus_time / session_duration) * 100
            
            stats_y = h - 10
            stats_text = f"Focus: {focus_percent:.1f}% | Distractions: {detector.distraction_count}"
            cv2.putText(frame, stats_text, (10, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        
        # display controls
        controls_y = h - 40
        controls_text = "q: Quit | r: Reset | c: Recalibrate"
        cv2.putText(frame, controls_text, (10, controls_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # show video status if active
        if detector.video_player and detector.video_player.is_playing:
            video_text = "VIDEO PLAYING - Focus to stop!"
            text_size = cv2.getTextSize(video_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.rectangle(frame, (text_x-10, 150), (text_x+text_size[0]+10, 180), (0, 0, 0), -1)
            cv2.putText(frame, video_text, (text_x, 175), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        #frame counter for debugging
        cv2.putText(frame, f"Frame: {frame_count}", (w - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # show frame
        cv2.imshow("Phone/Screen Distraction Detector", frame)
        
        # keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if detector.video_player:
                detector.video_player.stop_video()
                time.sleep(0.1)
            break
        elif key == ord('r'):
            # Reset statistics
            detector.distraction_start_time = None
            detector.distraction_count = 0
            detector.total_distraction_time = 0
            detector.eye_state_buffer = []
            detector.eye_state_history = []
            detector.eye_close_start_time = None
            detector.extended_distraction_start = None
            detector.video_played_this_distraction = False
            if detector.video_player:
                detector.video_player.stop_video()
                time.sleep(0.1)
            session_start = time.time()
            print("Statistics reset!")
            # Force focus status
            detector.is_distracted = False
        elif key == ord('c'):
            # Recalibrate
            print("Starting recalibration... Look at screen center")
            
            # Stop video player first
            if detector.video_player:
                detector.video_player.stop_video()
                time.sleep(0.1)
            
            # Reset calibration state
            detector.is_calibrated = False
            detector.calibration_frames = []
            detector.distraction_start_time = None
            detector.is_distracted = False
            detector.video_played_this_distraction = False
            
            recalibrating = True
            recal_frames = 0
            
            while recalibrating:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # calibration instructions
                cv2.putText(frame, "RECALIBRATING: Look at screen center", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Frames collected: {recal_frames}/30", 
                           (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.circle(frame, (w//2, h//2), 10, (0, 255, 255), -1)
                
                if detector.calibrate(frame):
                    recal_frames = len(detector.calibration_frames)
                
                if detector.is_calibrated:
                    recalibrating = False
                    print("Recalibration complete!")
                    break
                
                cv2.imshow("Phone/Screen Distraction Detector", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    recalibrating = False
                    break
    
    # cleanup
    detector.video_player.stop_video()
    cap.release()
    cv2.destroyAllWindows()
    
    # generate report
    print("\n" + "="*60)
    print("SESSION REPORT")
    print("="*60)
    
    session_duration = time.time() - session_start
    if session_duration > 0:
        focus_time = session_duration - detector.total_distraction_time
        focus_percent = (focus_time / session_duration) * 100
        
        print(f"Total session time: {session_duration:.1f} seconds")
        print(f"Number of distractions: {detector.distraction_count}")
        print(f"Total distracted time: {detector.total_distraction_time:.1f} seconds")
        print(f"Focus percentage: {focus_percent:.1f}%")
        
        if detector.distraction_count > 0:
            avg_distraction = detector.total_distraction_time / detector.distraction_count
            print(f"Average distraction length: {avg_distraction:.1f} seconds")
    
    print("="*60)

if __name__ == "__main__":
    main()