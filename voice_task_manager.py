import sqlite3
import sys
import speech_recognition as sr
import pyttsx3
import datetime
import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass
import signal
from contextlib import contextmanager
import uuid
import os
import json
import threading


# Constants
DEFAULT_DB_FILE = "voice_tasks.db"
BACKUP_DB_FILE = "voice_tasks_backup.db"
LOG_FILE = "voice_task_manager.log"
MAX_BACKUPS = 5
CONFIG_FILE = "voice_assistant_config.json"


# Enums
class TaskStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VoiceCommandType(Enum):
    ADD_TASK = auto()
    UPDATE_TASK = auto()
    DELETE_TASK = auto()
    SEARCH_TASKS = auto()
    VIEW_TASKS = auto()
    SET_REMINDER = auto()
    UNKNOWN = auto()


# Data Structures
@dataclass
class Task:
    id: str
    title: str
    status: TaskStatus
    due_date: str
    priority: Priority
    reminder_set: bool
    created_at: str
    updated_at: str


@dataclass
class Reminder:
    id: str
    task_id: str
    reminder_time: str
    message: str
    notified: bool
    created_at: str


# Exceptions
class VoiceAssistantError(Exception):
    """Base exception for Voice Assistant"""


class TaskNotFoundError(VoiceAssistantError):
    """Task not found exception"""


class DatabaseError(VoiceAssistantError):
    """Database operation failed"""


class SpeechRecognitionError(VoiceAssistantError):
    """Speech recognition failed"""


class InvalidCommandError(VoiceAssistantError):
    """Invalid voice command"""


# Database Manager
class DatabaseManager:
    def __init__(self, db_file: str = DEFAULT_DB_FILE):
        self.db_file = db_file
        self.logger = logging.getLogger(__name__)
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize the database with required tables"""
        try:
            with self._get_connection() as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('pending', 'completed')),
                        due_date TEXT NOT NULL,
                        priority TEXT NOT NULL CHECK(priority IN ('low', 'medium', 'high')),
                        reminder_set INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reminders (
                        id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        reminder_time TEXT NOT NULL,
                        message TEXT NOT NULL,
                        notified INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE CASCADE
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_time ON reminders(reminder_time)")
        except sqlite3.Error as e:
            self.logger.critical(f"Failed to initialize database: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {str(e)}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    def backup_database(self) -> None:
        """Create a backup of the database"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.db_file}.backup_{timestamp}"
            
            with self._get_connection() as src_conn:
                with sqlite3.connect(backup_file) as dest_conn:
                    src_conn.backup(dest_conn)
            
            self._cleanup_old_backups()
            self.logger.info(f"Database backup created: {backup_file}")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create backup: {str(e)}")
            raise DatabaseError(f"Backup failed: {str(e)}")

    def _cleanup_old_backups(self) -> None:
        """Clean up old backup files"""
        import glob
        backups = sorted(glob.glob(f"{self.db_file}.backup_*"), reverse=True)
        for old_backup in backups[MAX_BACKUPS:]:
            try:
                os.remove(old_backup)
                self.logger.info(f"Removed old backup: {old_backup}")
            except OSError as e:
                self.logger.warning(f"Failed to remove old backup {old_backup}: {str(e)}")


# Task Repository
class TaskRepository:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def create_task(self, task_data: Dict) -> Task:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        
        task = Task(
            id=task_id,
            title=task_data.get("title"),
            status=TaskStatus.PENDING.value,
            due_date=task_data.get("due_date"),
            priority=task_data.get("priority", Priority.MEDIUM.value),
            reminder_set=False,
            created_at=now,
            updated_at=now
        )
        
        try:
            with self.db_manager._get_connection() as conn:
                conn.execute("""
                    INSERT INTO tasks 
                    (id, title, status, due_date, priority, reminder_set, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id, task.title, task.status, task.due_date, task.priority,
                    task.reminder_set, task.created_at, task.updated_at
                ))
                conn.commit()
            return task
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create task: {str(e)}")
            raise DatabaseError(f"Failed to create task: {str(e)}")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_task(row)
                return None
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get task {task_id}: {str(e)}")
            raise DatabaseError(f"Failed to get task: {str(e)}")

    def get_all_tasks(self) -> List[Task]:
        """Get all tasks"""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC")
                return [self._row_to_task(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get all tasks: {str(e)}")
            raise DatabaseError(f"Failed to get tasks: {str(e)}")

    def update_task(self, task_id: str, update_data: Dict) -> Optional[Task]:
        """Update a task"""
        try:
            with self.db_manager._get_connection() as conn:
                # Get existing task
                cursor = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                # Prepare update
                update_fields = []
                update_values = []
                for field, value in update_data.items():
                    if field in row.keys() and field not in ["id", "created_at"]:
                        update_fields.append(f"{field} = ?")
                        update_values.append(value)

                if not update_fields:
                    return None

                # Add updated_at timestamp
                update_fields.append("updated_at = ?")
                update_values.append(datetime.datetime.now().isoformat())

                # Execute update
                update_query = f"UPDATE tasks SET {', '.join(update_fields)} WHERE id = ?"
                update_values.append(task_id)
                conn.execute(update_query, update_values)
                conn.commit()

                # Return updated task
                cursor = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
                return self._row_to_task(cursor.fetchone())
        except sqlite3.Error as e:
            self.logger.error(f"Failed to update task {task_id}: {str(e)}")
            raise DatabaseError(f"Failed to update task: {str(e)}")

    def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            self.logger.error(f"Failed to delete task {task_id}: {str(e)}")
            raise DatabaseError(f"Failed to delete task: {str(e)}")

    def search_tasks(self, **filters) -> List[Task]:
        """Search tasks with filters"""
        try:
            query = "SELECT * FROM tasks WHERE 1=1"
            params = []
            
            for field, value in filters.items():
                if field == "title" and value:
                    query += f" AND {field} LIKE ?"
                    params.append(f"%{value}%")
                elif field in ["status", "priority"] and value:
                    query += f" AND {field} = ?"
                    params.append(value)
            
            query += " ORDER BY created_at DESC"
            
            with self.db_manager._get_connection() as conn:
                cursor = conn.execute(query, params)
                return [self._row_to_task(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            self.logger.error(f"Failed to search tasks: {str(e)}")
            raise DatabaseError(f"Failed to search tasks: {str(e)}")

    def _row_to_task(self, row) -> Task:
        """Convert a database row to a Task object"""
        return Task(
            id=row["id"],
            title=row["title"],
            status=row["status"],
            due_date=row["due_date"],
            priority=row["priority"],
            reminder_set=bool(row["reminder_set"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )


# Reminder Repository
class ReminderRepository:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def create_reminder(self, reminder_data: Dict) -> Reminder:
        """Create a new reminder"""
        reminder_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        
        reminder = Reminder(
            id=reminder_id,
            task_id=reminder_data.get("task_id"),
            reminder_time=reminder_data.get("reminder_time"),
            message=reminder_data.get("message"),
            notified=False,
            created_at=now
        )
        
        try:
            with self.db_manager._get_connection() as conn:
                conn.execute("""
                    INSERT INTO reminders 
                    (id, task_id, reminder_time, message, notified, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    reminder.id, reminder.task_id, reminder.reminder_time,
                    reminder.message, reminder.notified, reminder.created_at
                ))
                conn.commit()
                
                # Update task's reminder_set flag
                conn.execute("""
                    UPDATE tasks SET reminder_set = 1 WHERE id = ?
                """, (reminder.task_id,))
                conn.commit()
                
            return reminder
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create reminder: {str(e)}")
            raise DatabaseError(f"Failed to create reminder: {str(e)}")

    def get_due_reminders(self) -> List[Reminder]:
        """Get reminders that are due"""
        try:
            current_time = datetime.datetime.now().strftime("%H:%M")
            with self.db_manager._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM reminders 
                    WHERE reminder_time <= ? AND notified = 0
                    ORDER BY reminder_time
                """, (current_time,))
                return [self._row_to_reminder(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get due reminders: {str(e)}")
            raise DatabaseError(f"Failed to get due reminders: {str(e)}")

    def mark_reminder_notified(self, reminder_id: str) -> bool:
        """Mark a reminder as notified"""
        try:
            with self.db_manager._get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE reminders SET notified = 1 WHERE id = ?
                """, (reminder_id,))
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            self.logger.error(f"Failed to mark reminder {reminder_id} as notified: {str(e)}")
            raise DatabaseError(f"Failed to mark reminder as notified: {str(e)}")

    def _row_to_reminder(self, row) -> Reminder:
        """Convert a database row to a Reminder object"""
        return Reminder(
            id=row["id"],
            task_id=row["task_id"],
            reminder_time=row["reminder_time"],
            message=row["message"],
            notified=bool(row["notified"]),
            created_at=row["created_at"]
        )


# Voice Command Parser
class VoiceCommandParser:
    @staticmethod
    def parse_command(command: str) -> Tuple[VoiceCommandType, Dict]:
        """Parse voice command and return command type and parameters"""
        command = command.lower()
        
        # Add task pattern: "add task 'title' due by YYYY-MM-DD priority high"
        add_match = re.search(
            r'(add|create)\s+task\s+(.*?)\s+due\s+by\s+(\d{4}-\d{2}-\d{2})\s+priority\s+(low|medium|high)',
            command
        )
        if add_match:
            return (
                VoiceCommandType.ADD_TASK,
                {
                    "title": add_match.group(2),
                    "due_date": add_match.group(3),
                    "priority": add_match.group(4)
                }
            )
        
        # Update task pattern: "update task 12345 to completed"
        update_match = re.search(
            r'(update|mark)\s+task\s+([a-f0-9-]+)\s+(?:to|as)\s+(pending|completed)',
            command
        )
        if update_match:
            return (
                VoiceCommandType.UPDATE_TASK,
                {
                    "task_id": update_match.group(2),
                    "status": update_match.group(3)
                }
            )
        
        # Delete task pattern: "delete task 12345"
        delete_match = re.search(
            r'(delete|remove)\s+task\s+([a-f0-9-]+)',
            command
        )
        if delete_match:
            return (
                VoiceCommandType.DELETE_TASK,
                {
                    "task_id": delete_match.group(2)
                }
            )
        
        # Search tasks pattern: "search tasks for meeting"
        search_match = re.search(
            r'search\s+tasks?\s+for\s+(.*)',
            command
        )
        if search_match:
            return (
                VoiceCommandType.SEARCH_TASKS,
                {
                    "keyword": search_match.group(1)
                }
            )
        
        # View tasks pattern: "view all tasks"
        if re.search(r'view\s+(all\s+)?tasks?', command):
            return (VoiceCommandType.VIEW_TASKS, {})
        
        # Set reminder pattern: "set reminder for task 12345 at 14:30"
        reminder_match = re.search(
            r'set\s+reminder\s+for\s+task\s+([a-f0-9-]+)\s+at\s+(\d{2}:\d{2})',
            command
        )
        if reminder_match:
            return (
                VoiceCommandType.SET_REMINDER,
                {
                    "task_id": reminder_match.group(1),
                    "reminder_time": reminder_match.group(2)
                }
            )
        
        return (VoiceCommandType.UNKNOWN, {})


# Voice Assistant
class VoiceAssistant:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_speech_engine()
        
        # Initialize database components
        self.db_manager = DatabaseManager()
        self.task_repo = TaskRepository(self.db_manager)
        self.reminder_repo = ReminderRepository(self.db_manager)
        
        # Initialize command parser
        self.command_parser = VoiceCommandParser()
        
        # Start reminder checker thread
        self._start_reminder_checker()

    def _initialize_speech_engine(self) -> None:
        """Initialize text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            self.recognizer = sr.Recognizer()
            
            # Configure microphone for ambient noise adjustment
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            self.logger.critical(f"Failed to initialize speech engine: {str(e)}")
            raise VoiceAssistantError("Failed to initialize speech components")

    def speak(self, text: str) -> None:
        """Convert text to speech"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Text-to-speech error: {str(e)}")
            raise SpeechRecognitionError("Text-to-speech conversion failed")

    def listen(self) -> Optional[str]:
        """Listen for voice command"""
        try:
            with sr.Microphone() as source:
                self.logger.info("Listening for command...")
                print("Listening... (press Ctrl+C to stop)")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
            try:
                command = self.recognizer.recognize_google(audio) # type: ignore
                self.logger.info(f"Recognized command: {command}")
                print(f"You said: {command}")
                return command.lower()
            except sr.UnknownValueError:
                self.speak("Sorry, I did not understand that.")
                return None
            except sr.RequestError as e:
                self.logger.error(f"Speech recognition service error: {str(e)}")
                self.speak("Sorry, there is an issue with the speech service.")
                return None
        except Exception as e:
            self.logger.error(f"Listening error: {str(e)}")
            raise SpeechRecognitionError("Listening for command failed")

    def execute_command(self, command: str) -> None:
        """Execute voice command"""
        try:
            command_type, params = self.command_parser.parse_command(command)
            
            if command_type == VoiceCommandType.ADD_TASK:
                self._handle_add_task(params)
            elif command_type == VoiceCommandType.UPDATE_TASK:
                self._handle_update_task(params)
            elif command_type == VoiceCommandType.DELETE_TASK:
                self._handle_delete_task(params)
            elif command_type == VoiceCommandType.SEARCH_TASKS:
                self._handle_search_tasks(params)
            elif command_type == VoiceCommandType.VIEW_TASKS:
                self._handle_view_tasks()
            elif command_type == VoiceCommandType.SET_REMINDER:
                self._handle_set_reminder(params)
            else:
                self.speak("Sorry, I did not understand that command.")
        except Exception as e:
            self.logger.error(f"Command execution error: {str(e)}")
            self.speak("An error occurred while executing the command.")

    def _handle_add_task(self, params: Dict) -> None:
        """Handle add task command"""
        try:
            task = self.task_repo.create_task({
                "title": params["title"],
                "due_date": params["due_date"],
                "priority": params["priority"]
            })
            self.speak(f"Task '{task.title}' added successfully with ID {task.id}")
        except Exception as e:
            self.logger.error(f"Add task error: {str(e)}")
            self.speak("Failed to add task. Please try again.")

    def _handle_update_task(self, params: Dict) -> None:
        """Handle update task command"""
        try:
            task = self.task_repo.update_task(params["task_id"], {
                "status": params["status"]
            })
            if task:
                self.speak(f"Task {task.id} updated to status {task.status}")
            else:
                self.speak("Task not found.")
        except Exception as e:
            self.logger.error(f"Update task error: {str(e)}")
            self.speak("Failed to update task. Please try again.")

    def _handle_delete_task(self, params: Dict) -> None:
        """Handle delete task command"""
        try:
            if self.task_repo.delete_task(params["task_id"]):
                self.speak("Task deleted successfully.")
            else:
                self.speak("Task not found.")
        except Exception as e:
            self.logger.error(f"Delete task error: {str(e)}")
            self.speak("Failed to delete task. Please try again.")

    def _handle_search_tasks(self, params: Dict) -> None:
        """Handle search tasks command"""
        try:
            tasks = self.task_repo.search_tasks(title=params["keyword"])
            if tasks:
                response = "Found tasks: " + ", ".join([f"{task.title} (ID: {task.id})" for task in tasks])
                self.speak(response)
            else:
                self.speak("No tasks found matching your search.")
        except Exception as e:
            self.logger.error(f"Search tasks error: {str(e)}")
            self.speak("Failed to search tasks. Please try again.")

    def _handle_view_tasks(self) -> None:
        """Handle view tasks command"""
        try:
            tasks = self.task_repo.get_all_tasks()
            if tasks:
                response = "Your tasks: " + ", ".join([
                    f"{task.title} (ID: {task.id}, Status: {task.status})" 
                    for task in tasks
                ])
                self.speak(response)
            else:
                self.speak("You have no tasks.")
        except Exception as e:
            self.logger.error(f"View tasks error: {str(e)}")
            self.speak("Failed to retrieve tasks. Please try again.")

    def _handle_set_reminder(self, params: Dict) -> None:
        """Handle set reminder command"""
        try:
            task = self.task_repo.get_task(params["task_id"])
            if not task:
                self.speak("Task not found.")
                return
                
            # Validate time format
            try:
                datetime.datetime.strptime(params["reminder_time"], "%H:%M")
            except ValueError:
                self.speak("Invalid time format. Please use HH:MM.")
                return
                
            reminder = self.reminder_repo.create_reminder({
                "task_id": params["task_id"],
                "reminder_time": params["reminder_time"],
                "message": f"Reminder for task: {task.title}"
            })
            self.speak(f"Reminder set for {params['reminder_time']} for task {task.id}")
        except Exception as e:
            self.logger.error(f"Set reminder error: {str(e)}")
            self.speak("Failed to set reminder. Please try again.")

    def _start_reminder_checker(self) -> None:
        """Start background thread to check for due reminders"""
        def reminder_checker():
            while True:
                try:
                    due_reminders = self.reminder_repo.get_due_reminders()
                    for reminder in due_reminders:
                        self.speak(reminder.message)
                        self.reminder_repo.mark_reminder_notified(reminder.id)
                except Exception as e:
                    self.logger.error(f"Reminder checker error: {str(e)}")
                time.sleep(60)  # Check every minute

        thread = threading.Thread(target=reminder_checker, daemon=True)
        thread.start()

    def run(self) -> None:
        """Run the voice assistant"""
        try:
            self.speak("Voice task manager initialized. How can I help you?")
            while True:
                try:
                    command = self.listen()
                    if command:
                        self.execute_command(command)
                except KeyboardInterrupt:
                    self.speak("Goodbye!")
                    break
                except Exception as e:
                    self.logger.error(f"Main loop error: {str(e)}")
                    self.speak("An error occurred. Please try again.")
        except Exception as e:
            self.logger.critical(f"Application error: {str(e)}")
            raise VoiceAssistantError("Application failed")


# Configuration and Setup
def setup_logging() -> None:
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )


def handle_signal(signum, frame) -> None:
    """Handle system signals"""
    logging.info(f"Received signal {signum}. Shutting down gracefully.")
    sys.exit(0)


def main() -> None:
    """Main application entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Configure logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting Voice Task Manager")
        
        # Run the voice assistant
        assistant = VoiceAssistant()
        assistant.run()

    except Exception as e:
        logger.critical(f"Application failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
