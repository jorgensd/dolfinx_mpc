// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT
//
// The "3D" button of a gallery card swaps the picture for the interactive scene of the demo, and
// back. The scene is only loaded when asked for, so the front page stays light.

document.addEventListener("click", (event) => {
  const button = event.target.closest(".gallery-3d");
  if (button === null) {
    return;
  }
  // The card is a link: the button must not follow it
  event.preventDefault();
  event.stopPropagation();
  const figure = button.closest(".gallery-figure");
  const picture = figure.querySelector("img");
  const scene = figure.querySelector("iframe");
  if (scene !== null) {
    scene.remove();
    picture.hidden = false;
    button.textContent = "3D";
    button.setAttribute("aria-pressed", "false");
    return;
  }
  const frame = document.createElement("iframe");
  frame.src = button.dataset.scene;
  frame.title = `${picture.alt}, interactive`;
  // The scene takes the place of the picture, at its size
  frame.style.aspectRatio = `${picture.naturalWidth || 4} / ${picture.naturalHeight || 3}`;
  picture.hidden = true;
  figure.insertBefore(frame, button);
  button.textContent = "2D";
  button.setAttribute("aria-pressed", "true");
});
