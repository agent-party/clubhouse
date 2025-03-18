"""
CLI output formatter for Kafka CLI.
"""

class CLIFormatter:
    """Format output for the CLI with colors and styling."""
    
    COLORS = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
    }
    
    STYLES = {
        "bold": "\033[1m",
        "underline": "\033[4m",
        "italic": "\033[3m",
    }
    
    RESET = "\033[0m"
    
    def _format_text(self, text: str, color: str = None, style: str = None) -> str:
        """
        Format text with color and style.
        
        Args:
            text: Text to format
            color: Color to use (from COLORS)
            style: Style to use (from STYLES)
            
        Returns:
            Formatted text
        """
        result = ""
        
        if color and color in self.COLORS:
            result += self.COLORS[color]
            
        if style and style in self.STYLES:
            result += self.STYLES[style]
            
        result += str(text)
        result += self.RESET
        
        return result
    
    def print_header(self, text: str) -> None:
        """
        Print header text.
        
        Args:
            text: Header text
        """
        print(f"\n{self._format_text(text, 'bright_cyan', 'bold')}")
        print(f"{self._format_text('=' * len(text), 'bright_cyan')}")
    
    def print_info(self, text: str) -> None:
        """
        Print information text.
        
        Args:
            text: Info text
        """
        print(text)
    
    def print_success(self, text: str) -> None:
        """
        Print success text.
        
        Args:
            text: Success text
        """
        print(self._format_text(f"✓ {text}", "green"))
    
    def print_error(self, text: str) -> None:
        """
        Print error text.
        
        Args:
            text: Error text
        """
        print(self._format_text(f"✗ {text}", "red", "bold"))
    
    def print_warning(self, text: str) -> None:
        """
        Print warning text.
        
        Args:
            text: Warning text
        """
        print(self._format_text(f"⚠ {text}", "yellow"))
    
    def print_command_help(self, command: str, description: str) -> None:
        """
        Print command help.
        
        Args:
            command: Command name
            description: Command description
        """
        print(f"{self._format_text(command, 'blue', 'bold')}: {description}")
    
    def print_agent_message(self, agent_name: str, message: str) -> None:
        """
        Print agent message.
        
        Args:
            agent_name: Agent name
            message: Message text
        """
        print(f"\n{self._format_text(agent_name, 'cyan', 'bold')}: {message}")
