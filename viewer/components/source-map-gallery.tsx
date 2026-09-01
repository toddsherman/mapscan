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
  const activeMap = activeIndex === null ? null : maps[activeIndex];

  useEffect(() => {
    if (activeIndex === null) return;

    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setActiveIndex(null);
    }

    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("keydown", closeOnEscape);
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, [activeIndex]);

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
              <Image
                src={activeMap.preview}
                alt={activeMap.alt}
                width={activeMap.width}
                height={activeMap.height}
                sizes="94vw"
                priority
              />
            </div>
            <footer>
              <span>
                {activeMap.width.toLocaleString()} × {activeMap.height.toLocaleString()} px
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
