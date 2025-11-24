"""Tool registry with validation and safety checks."""

import re
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator


class ToolSchema(BaseModel):
    """Schema for a tool definition."""
    
    name: str = Field(..., description="Tool name (must be alphanumeric with underscores)")
    description: str = Field(..., description="Tool description")
    input_schema: dict[str, Any] = Field(..., description="Pydantic-compatible input schema")
    output_schema: dict[str, Any] = Field(default_factory=dict, description="Expected output schema")
    timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0, description="Timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=5, description="Maximum retry attempts")
    executor: Optional[Callable] = Field(default=None, description="Tool execution function")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate tool name format."""
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError("Tool name must be alphanumeric with underscores, starting with letter or underscore")
        return v


class ToolRegistry:
    """Registry for managing and validating tools."""
    
    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, ToolSchema] = {}
        self._banned_patterns = [
            r'\bimport\s+',
            r'\bfrom\s+\w+\s+import\s+',
            r'\bopen\s*\(',
            r'\bexec\s*\(',
            r'\beval\s*\(',
            r'\bcompile\s*\(',
            r'\b__import__\s*\(',
            r'\binput\s*\(',
            r'\braw_input\s*\(',
            r'file\s*\(',
            r'execfile\s*\(',
        ]
    
    def register(self, tool: ToolSchema) -> None:
        """Register a tool in the registry."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def register_simple(
        self,
        name: str,
        description: str,
        executor: Callable,
        input_schema: Optional[dict[str, Any]] = None,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
    ) -> None:
        """Register a tool with a simple interface."""
        schema = ToolSchema(
            name=name,
            description=description,
            input_schema=input_schema or {},
            executor=executor,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        self.register(schema)
    
    def is_registered(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools
    
    def get_tool(self, tool_name: str) -> Optional[ToolSchema]:
        """Get a tool schema by name."""
        return self._tools.get(tool_name)
    
    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def validate_input(self, tool_name: str, input_data: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate input against tool schema."""
        tool = self.get_tool(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' is not registered"
        
        try:
            # Create a Pydantic model from the input schema
            InputModel = self._create_model_from_schema(tool.input_schema, f"{tool_name}Input")
            InputModel(**input_data)
            return True, None
        except ValidationError as e:
            return False, f"Input validation failed: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def validate_output(self, tool_name: str, output_data: Any) -> tuple[bool, Optional[str]]:
        """Validate output against tool schema."""
        tool = self.get_tool(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' is not registered"
        
        if not tool.output_schema:
            return True, None  # No output schema defined, accept anything
        
        try:
            OutputModel = self._create_model_from_schema(tool.output_schema, f"{tool_name}Output")
            if isinstance(output_data, dict):
                OutputModel(**output_data)
            else:
                OutputModel(result=output_data)
            return True, None
        except ValidationError as e:
            return False, f"Output validation failed: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def check_safety(self, tool_name: str, input_data: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Check if tool call is safe."""
        # Check if tool is registered
        if not self.is_registered(tool_name):
            return False, f"Tool '{tool_name}' is not in registry"
        
        # Check input for dangerous patterns
        input_str = str(input_data).lower()
        for pattern in self._banned_patterns:
            if re.search(pattern, input_str):
                return False, f"Input contains banned pattern: {pattern}"
        
        # Check for suspicious input patterns
        suspicious_patterns = [
            r'\.\./',  # Path traversal
            r'rm\s+-rf',  # Dangerous commands
            r'del\s+/',  # Windows delete
            r'format\s+',  # Format disk
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, input_str):
                return False, f"Input contains suspicious pattern: {pattern}"
        
        return True, None
    
    def _create_model_from_schema(self, schema: dict[str, Any], model_name: str) -> type[BaseModel]:
        """Create a Pydantic model from a JSON schema."""
        # Simple schema to Pydantic model conversion
        # For complex schemas, you might want to use pydantic's TypeAdapter or jsonschema-to-pydantic
        
        # Build annotations and field defaults
        annotations = {}
        field_defaults = {}
        
        if 'properties' in schema:
            for field_name, field_def in schema['properties'].items():
                field_type = self._get_pydantic_type(field_def.get('type', 'string'))
                annotations[field_name] = field_type
                if 'default' in field_def:
                    field_defaults[field_name] = field_def['default']
        
        # Create model class dynamically
        namespace = {
            '__annotations__': annotations,
            **field_defaults,
        }
        DynamicModel = type(model_name, (BaseModel,), namespace)
        return DynamicModel
    
    def _get_pydantic_type(self, json_type: str) -> type:
        """Convert JSON schema type to Python type."""
        type_map = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
        }
        return type_map.get(json_type, str)

