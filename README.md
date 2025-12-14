# OTF2 Timeline PDF Generator

A Python tool to generate a multi-page timeline PDF from an OTF2 trace
(e.g. recorded with Score-P).

The generated PDF combines:

- Overview of all MPI rank + thread lanes
- Focused depth view of a single lane (Enter/Leave nesting)
- Automatically paginated legend for all regions

This tool is designed for large-scale MPI/OpenMP applications, where
interactive GUI viewers become impractical for long time ranges or many threads.

---

## Features

- Rank + Thread Lane View
  - Timeline of all MPI ranks and threads
  - Automatic lane thinning when too many lanes are present
- Focused Depth View
  - Detailed Enter/Leave nesting for a selected lane
  - Function labels are drawn only when the visible width is sufficient
- Multi-range PDF Output
  - Overview page plus equally split subranges
  - Manual time ranges are also supported
- Auto-paginated Legend
  - Region names and colors split across pages automatically
- Scalable Rendering
  - Chunked drawing using PatchCollection for large traces

---

## Requirements

- Python 3.9 or later
- Python packages:
  - pandas
  - matplotlib
  - otf2 (Python bindings providing otf2.reader.Reader)

Example installation:

    pip install pandas matplotlib
    # Install otf2 according to your environment

---

## Usage

### Default (overview + automatic splits)

    python3 otf2_timeline_pdf.py traces.otf2 -o timeline.pdf

### Overview only

    python3 otf2_timeline_pdf.py traces.otf2 --auto-split 1

### Manually specified time ranges

    python3 otf2_timeline_pdf.py traces.otf2 \
      --ranges 0,120 0,10 0,5 \
      -o timeline.pdf

### Specify focused lane (depth view)

    python3 otf2_timeline_pdf.py traces.otf2 --focus r0:t0

---

## Output Layout

### Timeline Pages

Top panel:
- Timeline of all rank/thread lanes
- When the number of lanes exceeds --max-lane-rows,
  lanes are evenly thinned and only a subset is shown

Bottom panel:
- Depth view of the focused lane
- Enter/Leave nesting is shown vertically
- Function labels are drawn only if the visible width exceeds a threshold

### Legend Pages

- List of all regions with corresponding colors
- Automatically split across multiple pages if needed

---

## Important Options

| Option | Description |
|--------|-------------|
| --auto-split N | N=1: overview only; N>=2: overview + N equal subranges |
| --ranges | Manually specify time ranges (tmin,tmax in seconds) |
| --focus | Focused lane for depth view (e.g. r0:t0) |
| --max-lane-rows | Maximum number of rows in the lane overview |
| --max-depth | Maximum nesting depth shown in the depth view |
| --min-duration-ms | Drop events shorter than this duration |
| --chunk-size | Number of rectangles drawn per batch (performance tuning) |

---

## Implementation Notes

- Enter/Leave events are converted into intervals using a per-lane stack
- MPI rank is inferred from the OTF2 LocationGroup
  (multiple attribute names are tried to absorb binding differences)
- Rendering performance is improved using
  matplotlib.collections.PatchCollection with chunked drawing
- Lane thinning uses even sampling while always keeping the first and last lanes

---

## Limitations

- The entire trace is loaded into memory; extremely large traces may take time
- Some behavior depends on the specific OTF2 Python binding implementation
- GPU timelines or non-Enter/Leave events are not currently supported

---

## License

MIT License
