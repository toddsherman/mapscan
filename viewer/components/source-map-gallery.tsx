"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";

export type SourceMap = {
  title: string;
  shortTitle: string;
  preview: string;
  original: string;
  width: number;
  height: number;
  alt: string;
};

export function SourceMapGallery({ maps }: { maps: SourceMap[] }) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const isOpen = activeIndex !== null;
  const activeMap = activeIndex === null ? null : maps[activeIndex];
  const currentIndex = activeIndex ?? 0;
  const previousIndex = maps.length === 0
    ? 0
    : (currentIndex - 1 + maps.length) % maps.length;
  const nextIndex = maps.length === 0 ? 0 : (currentIndex + 1) % maps.length;

  function showRelativeMap(offset: number) {
    setActiveIndex((currentIndex) => {
      if (currentIndex === null || maps.length === 0) return currentIndex;
      return (currentIndex + offset + maps.length) % maps.length;
    });
  }

  useEffect(() => {
    if (!isOpen) return;

    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setActiveIndex(null);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        setActiveIndex((currentIndex) =>
          currentIndex === null
            ? null
            : (currentIndex - 1 + maps.length) % maps.length,
        );
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setActiveIndex((currentIndex) =>
          currentIndex === null ? null : (currentIndex + 1) % maps.length,
        );
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, [isOpen, maps.length]);

  return (
    <>
      <div className="story-source-grid" aria-label="The nine source maps">
        {maps.map((map, index) => (
          <button
            className="story-source-card"
            type="button"
            key={map.shortTitle}
            onClick={() => setActiveIndex(index)}
            aria-label={`Open ${map.title}`}
          >
            <span className="story-source-image">
              <Image
                src={map.preview}
                alt=""
                fill
                sizes="(max-width: 620px) 42vw, (max-width: 980px) 28vw, 230px"
              />
            </span>
            <span className="story-source-label">
              <span>{String(index + 1).padStart(2, "0")}</span>
              {map.shortTitle}
            </span>
          </button>
        ))}
      </div>

      {activeMap ? (
        <div
          className="story-lightbox"
          role="presentation"
          onMouseDown={() => setActiveIndex(null)}
        >
          <section
            className="story-lightbox-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={activeMap.title}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header>
              <div>
                <p className="story-kicker">Original source</p>
                <h2>{activeMap.title}</h2>
              </div>
              <button
                ref={closeButtonRef}
                type="button"
                onClick={() => setActiveIndex(null)}
                aria-label="Close source map"
              >
                ×
              </button>
            </header>
            <div className="story-lightbox-image">
              <button
                className="story-lightbox-nav story-lightbox-nav-previous"
                type="button"
                onClick={() => showRelativeMap(-1)}
                aria-label={`View previous source map: ${maps[previousIndex]?.title ?? "source map"}`}
              >
                <svg aria-hidden="true" viewBox="0 0 20 20">
                  <path d="m12.5 4.5-5 5.5 5 5.5" />
                </svg>
              </button>
              <Image
                src={activeMap.preview}
                alt={activeMap.alt}
                width={activeMap.width}
                height={activeMap.height}
                sizes="94vw"
                priority
              />
              <button
                className="story-lightbox-nav story-lightbox-nav-next"
                type="button"
                onClick={() => showRelativeMap(1)}
                aria-label={`View next source map: ${maps[nextIndex]?.title ?? "source map"}`}
              >
                <svg aria-hidden="true" viewBox="0 0 20 20">
                  <path d="m7.5 4.5 5 5.5-5 5.5" />
                </svg>
              </button>
            </div>
            <footer>
              <span>
                {currentIndex + 1} of {maps.length} · {activeMap.width.toLocaleString()} × {activeMap.height.toLocaleString()} px
              </span>
              <a href={activeMap.original} target="_blank" rel="noreferrer">
                Open full resolution ↗
              </a>
            </footer>
          </section>
        </div>
      ) : null}
    </>
  );
}
