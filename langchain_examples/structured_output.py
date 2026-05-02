"""
Structured output using Langchain.

Demonstrates the three distinct schema types for with_structured_output:
  1. Pydantic    - richest feature set; returns a validated Pydantic instance
  2. TypedDict   - lightweight typed dict; returns a plain dict, no validation
  3. JSON Schema - raw dict schema; maximum flexibility, returns a plain dict
"""

from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from config.llm_model import LLM_MODEL

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------


class MovieReview(BaseModel):
    """A structured review of a movie."""

    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")


class BookSummary(TypedDict):
    """A brief summary of a book."""

    title: str
    author: str
    genre: str
    year_published: int


class BookSummaryModel(BaseModel):
    """Pydantic schema used to validate a BookSummary dict."""

    title: str = Field(description="The title of the book")
    author: str = Field(description="The author of the book")
    genre: str = Field(description="The literary genre")
    year_published: int = Field(
        description="Year the book was published", ge=1, le=2100
    )


# JSON Schema expressed as a plain Python dict - no import needed
SONG_JSON_SCHEMA = {
    "title": "SongDetails",
    "description": "Key details about a song.",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Title of the song"},
        "artist": {"type": "string", "description": "The performing artist"},
        "year": {"type": "integer", "description": "Release year"},
        "genre": {"type": "string", "description": "Musical genre"},
    },
    "required": ["title", "artist", "year", "genre"],
}


class SongDetailsModel(BaseModel):
    """Pydantic schema used to validate a SONG_JSON_SCHEMA dict."""

    title: str = Field(description="Title of the song")
    artist: str = Field(description="The performing artist")
    year: int = Field(description="Release year", ge=1900, le=2100)
    genre: str = Field(description="Musical genre")


def demonstrate_pydantic(llm: BaseChatModel) -> MovieReview:
    """Way 1: Pydantic BaseModel structured output."""
    print("WAY 1: Pydantic BaseModel")
    print("=" * 50)

    llm_pydantic = llm.with_structured_output(MovieReview)
    print(f"Return type               : {type(llm_pydantic)}")
    print(f"LLM with structured output: {llm_pydantic}")
    print("=" * 50)
    movie: MovieReview = llm_pydantic.invoke(
        "Give me a structured review of the movie Inception."
    )

    # The return type is a Pydantic model instance, not a plain dict
    print(f"Return type      : {type(movie)}")
    print(f"Is Pydantic model: {isinstance(movie, BaseModel)}")
    print("=" * 50)

    # Individual fields are fully typed and accessible as attributes
    print(f"title    : {movie.title}")
    print(f"year     : {movie.year}")
    print(f"director : {movie.director}")
    print("=" * 50)

    # Pydantic enforces constraints at instantiation time; le=10.0 is violated
    try:
        MovieReview(
            title="Test",
            year=2024,
            director=2,
        )
    except ValidationError as exc:
        print("Pydantic caught invalid director (2 is not a string):")
        print(exc.errors()[0]["msg"])
    print("=" * 50)

    # include_raw=True surfaces the raw AIMessage alongside the parsed output;
    # useful for accessing response metadata (token usage, model name, etc.)
    llm_raw = llm.with_structured_output(MovieReview, include_raw=True)
    print(f"Return type               : {type(llm_raw)}")
    print(f"LLM with structured output (include_raw=True): {llm_raw}")
    print("=" * 50)

    raw_result: dict = llm_raw.invoke(
        "Give me a structured review of the movie The Matrix."
    )

    print("include_raw=True keys  :", list(raw_result.keys()))
    print("parsed type            :", type(raw_result["parsed"]))
    print("raw AIMessage type     :", type(raw_result["raw"]))
    print("parsing_error          :", raw_result["parsing_error"])
    print("=" * 50)

    return movie


def demonstrate_typeddict(llm: BaseChatModel) -> BookSummary:
    """Way 2: TypedDict structured output."""
    print("WAY 2: TypedDict")
    print("=" * 50)

    llm_typed = llm.with_structured_output(BookSummary)
    print(f"Return type               : {type(llm_typed)}")
    print(f"LLM with structured output: {llm_typed}")
    print("=" * 50)

    book: BookSummary = llm_typed.invoke(
        "Summarise the book 1984 by George Orwell."
    )

    # TypedDict returns a plain dict at runtime despite the annotation
    print(f"Return type   : {type(book)}")
    print(f"Is plain dict : {isinstance(book, dict)}")
    print("=" * 50)

    # Fields are accessed as dict keys, not attributes
    print(f"title          : {book['title']}")
    print(f"author         : {book['author']}")
    print(f"genre          : {book['genre']}")
    print(f"year_published : {book['year_published']}")
    print("=" * 50)

    # __annotations__ shows the keys the model was told to populate
    print("TypedDict annotations:", BookSummary.__annotations__)
    print("=" * 50)

    # TypedDict provides no runtime validation; manually validate the dict
    # by constructing a Pydantic model from it.  A ValidationError is raised
    # if any field is missing or its value violates a constraint.
    print("Manual Pydantic validation of the returned dict:")
    try:
        validated_book = BookSummaryModel(**book)
        print(f"Validated type      : {type(validated_book)}")
        print(f"Is Pydantic instance: {isinstance(validated_book, BaseModel)}")
        print(f"year_published valid: {validated_book.year_published}")
    except ValidationError as exc:
        print(f"Validation failed: {exc}")
    print("=" * 50)

    return book


def demonstrate_json_schema(llm: BaseChatModel) -> dict:
    """Way 3: JSON Schema (plain dict) structured output."""
    print("WAY 3: JSON Schema (plain dict)")
    print("=" * 50)

    llm_json = llm.with_structured_output(SONG_JSON_SCHEMA)
    print(f"Return type               : {type(llm_json)}")
    print(f"LLM with structured output: {llm_json}")
    print("=" * 50)

    song: dict = llm_json.invoke(
        "Give me details about the song Bohemian Rhapsody by Queen."
    )

    # Like TypedDict, the result is a plain dict
    print(f"Return type  : {type(song)}")
    print(f"Is plain dict: {isinstance(song, dict)}")
    print("=" * 50)

    # The schema itself is just a dict - no Python class required
    print("Schema title    :", SONG_JSON_SCHEMA["title"])
    print("Required fields :", SONG_JSON_SCHEMA["required"])
    print("=" * 50)

    print(f"title  : {song['title']}")
    print(f"artist : {song['artist']}")
    print(f"year   : {song['year']}")
    print(f"genre  : {song['genre']}")
    print("=" * 50)

    # JSON Schema provides no runtime validation; manually validate the dict
    # by constructing a Pydantic model.  The ge/le constraints on `year` will
    # catch values outside [1900, 2100] that the raw schema cannot enforce.
    print("Manual Pydantic validation of the returned dict:")
    try:
        validated_song = SongDetailsModel(**song)
        print(f"Validated type      : {type(validated_song)}")
        print(f"Is Pydantic instance: {isinstance(validated_song, BaseModel)}")
    except ValidationError as exc:
        print(f"Validation failed: {exc}")
    print("=" * 50)

    return song


def run(llm: BaseChatModel) -> None:
    """Run the structured output example."""
    movie = demonstrate_pydantic(llm)
    book = demonstrate_typeddict(llm)
    song = demonstrate_json_schema(llm)

    # Summary: key differences at a glance
    print("SUMMARY - return types per schema approach")
    print("=" * 50)
    pydantic_label = f"{type(movie).__name__:12}"
    typeddict_label = f"{type(book).__name__:12}"
    jsonschema_label = f"{type(song).__name__:12}"
    print(f"Pydantic BaseModel : {pydantic_label} | validated instance")
    print(
        f"TypedDict          : {typeddict_label} | plain dict, no validation"
    )
    print(
        f"JSON Schema        : {jsonschema_label} | plain dict, no validation"
    )
    print("=" * 50)


if __name__ == "__main__":
    run(LLM_MODEL)
