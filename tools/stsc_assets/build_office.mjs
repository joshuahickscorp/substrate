import fs from "node:fs/promises";
import path from "node:path";
import {
  Presentation,
  PresentationFile,
  SpreadsheetFile,
  Workbook,
  layers,
  table,
  text,
} from "@oai/artifact-tool";

const outputRoot = path.resolve(process.argv[2]);

async function writeBlob(filename, blob) {
  await fs.writeFile(filename, new Uint8Array(await blob.arrayBuffer()));
}

async function buildWorkbook() {
  const workbook = Workbook.create();
  const raw = workbook.worksheets.add("Raw Measurements");
  const calculations = workbook.worksheets.add("Calculations");
  const results = workbook.worksheets.add("Results");

  raw.showGridLines = false;
  raw.freezePanes.freezeRows(1);
  raw.getRange("A1:F7").values = [
    ["Timestamp", "Sensor", "Device Count", "Ledger Count", "Temperature C", "Flag"],
    [new Date("2026-07-28T14:27:00Z"), "north-1", 12840, 12690, 23.4, ""],
    [new Date("2026-07-28T14:32:00Z"), "north-1", 13220, 12257, 23.6, "incident"],
    [new Date("2026-07-28T14:37:00Z"), "north-1", 13610, 12916, 23.8, "restart"],
    [new Date("2026-07-28T14:42:00Z"), "north-1", 14002, 13315, 23.7, ""],
    [new Date("2026-07-28T14:47:00Z"), "north-1", 14395, 13847, 23.5, ""],
    [new Date("2026-07-28T14:52:00Z"), "north-1", 14788, 14276, 23.3, ""],
  ];
  raw.getRange("A1:F1").format = {
    fill: "#0B2545",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
  };
  raw.getRange("A2:A7").format.numberFormat = "yyyy-mm-dd hh:mm";
  raw.getRange("C2:D7").format.numberFormat = "#,##0";
  raw.getRange("E2:E7").format.numberFormat = "0.0";
  raw.getRange("A1:F7").format.borders = {
    insideHorizontal: { style: "thin", color: "#D9DEE5" },
    bottom: { style: "thin", color: "#AAB4C0" },
  };
  raw.getRange("A1:A7").format.columnWidth = 20;
  raw.getRange("B1:B7").format.columnWidth = 14;
  raw.getRange("C1:E7").format.columnWidth = 15;
  raw.getRange("F1:F7").format.columnWidth = 14;

  calculations.showGridLines = false;
  calculations.getRange("A1:E7").values = [
    ["Timestamp", "Device Count", "Ledger Count", "Absolute Difference", "Variance"],
    [null, null, null, null, null],
    [null, null, null, null, null],
    [null, null, null, null, null],
    [null, null, null, null, null],
    [null, null, null, null, null],
    [null, null, null, null, null],
  ];
  calculations.getRange("A2:C2").formulas = [[
    "='Raw Measurements'!A2",
    "='Raw Measurements'!C2",
    "='Raw Measurements'!D2",
  ]];
  calculations.getRange("A2:C7").fillDown();
  calculations.getRange("D2").formulas = [["=ABS(B2-C2)"]];
  calculations.getRange("D2:D7").fillDown();
  calculations.getRange("E2").formulas = [["=IF(B2=0,0,D2/B2)"]];
  calculations.getRange("E2:E7").fillDown();
  calculations.getRange("A1:E1").format = {
    fill: "#2E74B5",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
  };
  calculations.getRange("A2:A7").format.numberFormat = "yyyy-mm-dd hh:mm";
  calculations.getRange("B2:D7").format.numberFormat = "#,##0";
  calculations.getRange("E2:E7").format.numberFormat = "0.0%";
  calculations.getRange("A1:A7").format.columnWidth = 20;
  calculations.getRange("B1:D7").format.columnWidth = 18;
  calculations.getRange("E1:E7").format.columnWidth = 14;

  results.showGridLines = false;
  results.getRange("A1:F1").merge();
  results.getRange("A1").values = [["Project Aurora telemetry review"]];
  results.getRange("A1:F1").format = {
    fill: "#0B2545",
    font: { bold: true, color: "#FFFFFF", size: 18 },
  };
  results.getRange("A1:F1").format.rowHeight = 32;
  results.getRange("A3:B6").values = [
    ["Metric", "Value"],
    ["Maximum variance", null],
    ["Latest variance", null],
    ["Threshold", 0.045],
  ];
  results.getRange("B4").formulas = [["=MAX('Calculations'!E2:E7)"]];
  results.getRange("B5").formulas = [["='Calculations'!E7"]];
  results.getRange("B4:B6").format.numberFormat = "0.0%";
  results.getRange("A3:B3").format = {
    fill: "#E8EEF5",
    font: { bold: true, color: "#0B2545" },
  };
  results.getRange("A3:B6").format.borders = {
    preset: "outside",
    style: "thin",
    color: "#AAB4C0",
  };
  results.getRange("A1:A6").format.columnWidth = 24;
  results.getRange("B1:B6").format.columnWidth = 16;
  results.getRange("D3:E3").values = [["Timestamp", "Variance"]];
  results.getRange("D4:D9").values = [
    ["14:27"],
    ["14:32"],
    ["14:37"],
    ["14:42"],
    ["14:47"],
    ["14:52"],
  ];
  results.getRange("E4").formulas = [["='Calculations'!E2"]];
  results.getRange("E4:E9").fillDown();
  const chart = results.charts.add("line", results.getRange("D3:E9"));
  chart.title = "Observed telemetry variance";
  chart.hasLegend = false;
  chart.xAxis = { axisType: "textAxis", title: { text: "Time" } };
  chart.yAxis = { numberFormatCode: "0.0%", title: { text: "Variance" } };
  chart.setPosition("D2", "K18");

  await fs.mkdir(outputRoot, { recursive: true });
  for (const sheetName of ["Raw Measurements", "Calculations", "Results"]) {
    const preview = await workbook.render({
      sheetName,
      autoCrop: "all",
      scale: 2,
      format: "png",
    });
    await writeBlob(
      path.join(outputRoot, `xlsx-${sheetName.toLowerCase().replaceAll(" ", "-")}.png`),
      preview,
    );
  }
  const inspection = await workbook.inspect({
    kind: "table,formula",
    maxChars: 8000,
    tableMaxRows: 12,
    tableMaxCols: 8,
  });
  await fs.writeFile(path.join(outputRoot, "xlsx-inspection.ndjson"), inspection.ndjson);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    maxChars: 4000,
  });
  await fs.writeFile(path.join(outputRoot, "xlsx-errors.ndjson"), errors.ndjson);
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(path.join(outputRoot, "aurora-telemetry.xlsx"));
}

function addCover(presentation) {
  const slide = presentation.slides.add();
  slide.compose(
    layers({ name: "codex-grid-layout-library#slide-01", width: "fill", height: "fill" }, [
      text(["PROJECT AURORA"], {
        name: "eyebrow",
        position: { left: 41.33, top: 41.18 },
        width: 598.67,
        height: 68.15,
        style: { fontSize: "32px", typeface: "Helvetica Neue", color: "#000000" },
      }),
      text(["Recovery readiness"], {
        name: "title",
        position: { left: 41.33, top: 182.55 },
        width: 992,
        height: 261.57,
        style: {
          fontSize: "80px",
          typeface: "Helvetica Neue",
          color: "#000000",
          verticalAlignment: "bottom",
        },
      }),
      text(["Telemetry evidence and restart authority"], {
        name: "subtitle",
        position: { left: 41.33, top: 497.87 },
        width: 598.67,
        height: 113.41,
        style: { fontSize: "32px", typeface: "Helvetica Neue", color: "#000000" },
      }),
    ]),
    { frame: { left: 0, top: 0, width: 1280, height: 720 }, baseUnit: 1 },
  );
  slide.speakerNotes.textFrame.setText("[Sources]\nInternal STSC-1 generated telemetry fixture.");
}

function addEvidenceTable(presentation) {
  const slide = presentation.slides.add();
  slide.compose(
    layers({ name: "codex-grid-layout-library#slide-14", width: "fill", height: "fill" }, [
      text(["Variance recovered; integrity remains open"], {
        name: "title",
        position: { left: 41.33, top: 36.12 },
        width: 1197.33,
        height: 109.97,
        style: { fontSize: "38.67px", typeface: "Helvetica Neue", color: "#000000" },
      }),
      text(["EVIDENCE", "Two windows pass 4.5%; packet-digest reconciliation is still incomplete."], {
        name: "body",
        position: { left: 42.09, top: 111.02 },
        width: 1197.33,
        height: 106.27,
        style: { fontSize: "21.33px", typeface: "Helvetica Neue", color: "#000000" },
      }),
      table({
        name: "telemetry-table",
        rows: 7,
        columns: 5,
        values: [
          ["Time ET", "Device", "Ledger", "Difference", "Variance"],
          ["14:27", "12,840", "12,690", "150", "1.2%"],
          ["14:32", "13,220", "12,257", "963", "7.3%"],
          ["14:37", "13,610", "12,916", "694", "5.1%"],
          ["14:42", "14,002", "13,315", "687", "4.9%"],
          ["14:47", "14,395", "13,847", "548", "3.8%"],
          ["14:52", "14,788", "14,276", "512", "3.5%"],
        ],
        columnWidths: [250, 230, 230, 240, 247.33],
        position: { left: 41.33, top: 236.33 },
        width: 1197.33,
        height: 360,
      }),
      text(["2"], {
        name: "footer",
        position: { left: 1184.18, top: 659.24 },
        width: 54.48,
        height: 25.33,
        style: {
          fontSize: "13.33px",
          typeface: "Helvetica Neue",
          color: "#000000",
          alignment: "right",
        },
      }),
    ]),
    { frame: { left: 0, top: 0, width: 1280, height: 720 }, baseUnit: 1 },
  );
  slide.speakerNotes.textFrame.setText("[Sources]\nInternal STSC-1 generated telemetry fixture.");
}

function addClose(presentation) {
  const slide = presentation.slides.add();
  slide.compose(
    layers({ name: "codex-grid-layout-library#slide-26", width: "fill", height: "fill" }, [
      text(["NEXT"], {
        name: "eyebrow",
        position: { left: 41.33, top: 41.18 },
        width: 169.33,
        height: 68.15,
        style: { fontSize: "32px", typeface: "Helvetica Neue", color: "#000000" },
      }),
      text(["Verify twice, then restart"], {
        name: "title",
        position: { left: 41.33, top: 182.55 },
        width: 992,
        height: 261.57,
        style: {
          fontSize: "80px",
          typeface: "Helvetica Neue",
          color: "#000000",
          verticalAlignment: "bottom",
        },
      }),
      text(["Recalibrate sensors", "Reconcile raw packets", "Record the digest"], {
        name: "actions",
        position: { left: 41.33, top: 522.13 },
        width: 500,
        height: 113.41,
        style: { fontSize: "32px", typeface: "Helvetica Neue", color: "#000000" },
      }),
    ]),
    { frame: { left: 0, top: 0, width: 1280, height: 720 }, baseUnit: 1 },
  );
  slide.speakerNotes.textFrame.setText("[Sources]\nInternal STSC-1 generated telemetry fixture.");
}

async function buildPresentation() {
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  addCover(presentation);
  addEvidenceTable(presentation);
  addClose(presentation);
  for (const [index, slide] of presentation.slides.items.entries()) {
    const preview = await presentation.export({ slide, format: "png", scale: 2 });
    await writeBlob(path.join(outputRoot, `pptx-slide-${index + 1}.png`), preview);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(
      path.join(outputRoot, `pptx-slide-${index + 1}.layout.json`),
      await layout.text(),
    );
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(outputRoot, "pptx-montage.webp"), montage);
  const inspection = await presentation.inspect({
    kind: "slide,textbox,shape,table,notes,layout",
    maxChars: 12000,
  });
  await fs.writeFile(path.join(outputRoot, "pptx-inspection.ndjson"), inspection.ndjson);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(path.join(outputRoot, "aurora-recovery.pptx"));
}

await fs.mkdir(outputRoot, { recursive: true });
await buildWorkbook();
await buildPresentation();
