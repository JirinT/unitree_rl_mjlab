# Copyright 2025 Marc Duclusaud & Grégoire Passault

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import os


def error(message):
    print(message)


def bright(message):
    print("")
    print(message)


def run_or_fail(cmd):
    if os.system(cmd) != 0:
        die("Error while running " + cmd)


def emphasis(text):
    return text


def yellow(text):
    return text


def red(text):
    return text


def success(text):
    return text


def die(message):
    print(error(message))
    exit()


def print_parameter(name, value, infos=None, warning=None):
    result = f"- {success(name)}: {yellow(str(value))}"

    if infos is not None:
        result += f" ({infos})"
    if warning is not None:
        result += red(f" ({warning})")

    print(result)
