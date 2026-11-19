"""Generic decorator registry."""

from __future__ import annotations

from collections.abc import Callable


class Registry:
    def __init__(self, kind: str) -> None:
        """Start an empty registry labelled with the kind it holds.

        Args:
            kind: Noun used in error messages, e.g. model or benchmark.
        """
        self.kind = kind
        self._entries: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        """Build a class decorator that files a class under name.

        Args:
            name: Key the decorated class is registered under.

        Returns:
            A decorator that stores the class and returns it unchanged.
        """

        def decorate(cls: type) -> type:
            """Store the class under name, rejecting duplicate keys.

            Args:
                cls: Class to register.

            Returns:
                The same class, so the decorator is transparent.
            """
            if name in self._entries:
                raise KeyError(
                    f"{self.kind} {name!r} is already registered ({self._entries[name]})"
                )
            self._entries[name] = cls
            return cls

        return decorate

    def get(self, name: str) -> type:
        """Look up a registered class, listing the known keys on failure.

        Args:
            name: Key to look up.

        Returns:
            The class registered under name.
        """
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(sorted(self._entries)) or "<none>"
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {known}") from None

    def names(self) -> list[str]:
        """Report every registered key.

        Returns:
            The registered keys in alphabetical order.
        """
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        """Check whether a key has been registered.

        Args:
            name: Key to test.

        Returns:
            True if name is registered, False otherwise.
        """
        return name in self._entries
