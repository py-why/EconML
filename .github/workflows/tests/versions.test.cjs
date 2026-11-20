const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const {test} = require("node:test");
const {runInNewContext} = require("node:vm");

const code = readFileSync(join(__dirname, "..", "docs_assets", "versions.js"), "utf8");

class Element {
    constructor(tag = "div") {
        this.tagName = tag;
        this.children = [];
        this.dataset = {};
        this.attributes = {};
        this.classes = new Set();
        this.classList = {add: (...names) => names.forEach((name) => this.classes.add(name))};
    }

    get options() { return this.children; }
    append(...elements) {
        for (const element of elements) {
            if (element.parentElement) {
                const siblings = element.parentElement.children;
                siblings.splice(siblings.indexOf(element), 1);
            }
            this.children.push(element);
            element.parentElement = this;
        }
    }
    prepend(element) { this.append(element); this.children.unshift(this.children.pop()); }
    replaceChildren(...elements) { this.children = []; this.append(...elements); }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener(event, callback) { this.callback = callback; }
}

async function load({
    base = "https://example.test/EconML/",
    page = "v0.16.0/_autosummary/econml.dml.LinearDML.html#fit",
    currentVersion = "v0.16.0",
    themed = true,
    manifest,
    responseOK = true,
    reject = false,
    headStatus = 200,
    headReject = false,
}) {
    const select = new Element("select");
    select.hidden = true;
    const status = new Element();
    const panel = new Element("aside");
    panel.dataset.version = currentVersion;
    const currentLabel = new Element("strong");
    currentLabel.textContent = currentVersion === "prerelease"
        ? "Prerelease (not yet released) - source abcdef123456" : currentVersion;
    const elements = {select, '[role="status"]': status, strong: currentLabel, a: new Element("a"),
        label: new Element("label")};
    panel.append(...Object.values(elements));
    panel.querySelector = (selector) => elements[selector];
    const sidebar = new Element("nav");
    const main = new Element("main");
    const errors = [];
    const probes = [];
    const context = {
        URL,
        document: {
            currentScript: {src: `${base}_versions/versions.js`},
            getElementById() { return panel; },
            querySelector(selector) {
                if (!themed) { return null; }
                return selector === ".wy-nav-side" ? sidebar : main;
            },
            createElement(tag) { return new Element(tag); },
        },
        fetch: async (url, options) => {
            assert.equal(options.cache, "no-cache");
            if (options.method === "HEAD") {
                probes.push(url.href);
                if (headReject) { throw new Error("Offline"); }
                return {ok: headStatus >= 200 && headStatus < 300, status: headStatus};
            }
            assert.equal(url.href, `${base}versions.json`);
            if (reject) { throw new Error("Offline"); }
            return {ok: responseOK, status: 404, json: async () => manifest};
        },
        window: {location: {href: new URL(page, base).href, assign(url) { context.navigated = url; }}},
        console: {error(...args) { errors.push(args); }},
    };
    runInNewContext(code, context);
    await new Promise((resolve) => setImmediate(resolve));
    // The browser initializes select.value from the selected option.
    select.value = select.options.find((option) => option.selected)?.value;
    return {select, status, panel, sidebar, main, errors, probes, context};
}

function versions() {
    return {
        schema: 1,
        stable: "v0.17.1",
        versions: [
            {id: "prerelease", path: "prerelease/", label: "Prerelease (main)"},
            {id: "v0.17.1", path: "v0.17.1/", label: "0.17.1"},
            {id: "v0.16.0", path: "v0.16.0/", label: "0.16.0"},
        ],
    };
}

function clickEvent(options = {}) {
    return {button: 0, ...options, preventDefault() { this.defaultPrevented = true; }};
}

for (const base of ["https://www.pywhy.org/EconML/", "https://kbattocchi.github.io/EconML/", "https://example.test/"]) {
    test(`archived page sees new versions and keeps the same API page within ${base}`, async () => {
        const {select, errors, probes, context} = await load({base, manifest: versions()});
        assert.equal(select.hidden, false);
        assert.equal(select.options.length, 3);
        assert.equal(select.options[1].textContent, "0.17.1 (stable)");
        assert.equal(select.options[2].selected, true);
        select.value = select.options[0].value;
        await select.callback();
        assert.deepEqual(probes, [`${base}prerelease/_autosummary/econml.dml.LinearDML.html`]);
        assert.equal(context.navigated, `${base}prerelease/_autosummary/econml.dml.LinearDML.html#fit`);
        assert.equal(errors.length, 0);
    });
}

test("stable-root pages preserve their path, search query and anchor", async () => {
    const {select, context} = await load({
        manifest: versions(), currentVersion: "v0.17.1", page: "search.html?q=forest#results",
    });
    select.value = select.options[2].value;
    await select.callback();
    assert.equal(context.navigated, "https://example.test/EconML/v0.16.0/search.html?q=forest#results");
});

test("prerelease pages remove their mount prefix when switching to a release", async () => {
    const {select, context, main} = await load({
        manifest: versions(), currentVersion: "prerelease", page: "prerelease/spec/estimation/dml.html#example",
    });
    select.value = select.options[1].value;
    await select.callback();
    assert.equal(context.navigated, "https://example.test/EconML/v0.17.1/spec/estimation/dml.html#example");
    const banner = main.children[0];
    assert.equal(banner.className, "econml-docs-notice");
    assert.equal(banner.attributes.role, "note");
    assert.match(banner.textContent, /not yet released.*abcdef123456/);
    assert.equal(banner.children[0].href, "https://example.test/EconML/spec/estimation/dml.html#example");
});

for (const page of ["v0.16.0/", "v0.16.0/index.html", "v0.16.0/spec/"]) {
    test(`directory and landing-page paths switch correctly from ${page}`, async () => {
        const {select, context} = await load({manifest: versions(), page});
        select.value = select.options[0].value;
        await select.callback();
        const relative = page === "v0.16.0/spec/" ? "spec/index.html" : "index.html";
        assert.equal(context.navigated, `https://example.test/EconML/prerelease/${relative}`);
    });
}

test("only a missing page falls back to the version landing page without its obsolete fragment", async () => {
    const {select, probes, context} = await load({manifest: versions(), headStatus: 404});
    select.value = select.options[0].value;
    const switching = select.callback();
    assert.equal(select.disabled, true);
    await switching;
    assert.equal(probes.length, 1);
    assert.equal(context.navigated, "https://example.test/EconML/prerelease/index.html");
});

for (const options of [
    {headStatus: 500}, {headReject: true}, {page: "https://example.test/outside/index.html", themed: false},
]) {
    test(`failed existence check stays on the page and allows retry: ${JSON.stringify(options)}`, async () => {
        const {select, status, errors, context} = await load({manifest: versions(), ...options});
        select.value = select.options[0].value;
        await select.callback();
        assert.equal(context.navigated, undefined);
        assert.equal(select.disabled, false);
        assert.equal(select.value, "https://example.test/EconML/v0.16.0/index.html");
        assert.match(status.textContent, /Try again or use the all-versions link/);
        assert.equal(errors.length, 1);
    });
}

test("a colon in the source page name cannot become a destination URL scheme", async () => {
    const {select, context} = await load({manifest: versions(), page: "v0.16.0/https:example.html"});
    select.value = select.options[0].value;
    await select.callback();
    assert.equal(context.navigated, "https://example.test/EconML/prerelease/https:example.html");
});

test("theme pages get a native collapsible sidebar panel, not a content overlay", async () => {
    const {panel, sidebar, main, select} = await load({manifest: versions()});
    assert.equal(panel.parentElement, sidebar);
    assert.equal(sidebar.classes.has("econml-has-versions"), true);
    assert.equal(panel.classes.has("rst-versions"), true);
    const details = panel.children[0];
    assert.equal(details.tagName, "details");
    assert.equal(details.children[0].tagName, "summary");
    assert.equal(details.children[0].textContent, "Version: v0.16.0");
    assert.equal(select.parentElement.className, "rst-other-versions");
    assert.equal(main.children.length, 1);
});

test("archived releases identify the current stable release using shared manifest metadata", async () => {
    const {main} = await load({manifest: versions()});
    const banner = main.children[0];
    assert.equal(banner.attributes.role, "note");
    assert.match(banner.textContent, /0\.16\.0.*not the current stable release \(0\.17\.1\)/);
    assert.equal(banner.children[0].href,
        "https://example.test/EconML/_autosummary/econml.dml.LinearDML.html#fit");
});

for (const page of ["index.html", "v0.17.1/index.html"]) {
    test(`current stable docs do not get an outdated-version notice at ${page}`, async () => {
        const {main, select} = await load({manifest: versions(), currentVersion: "v0.17.1", page});
        assert.equal(main.children.length, 0);
        assert.equal(select.hidden, false);
    });
}

for (const currentVersion of ["prerelease", "v0.16.0"]) {
    test(`${currentVersion} banner keeps the corresponding stable page, query and current anchor`, async () => {
        const page = `${currentVersion}/_autosummary/econml.dml.LinearDML.html?q=forest#fit`;
        const {main, context, probes} = await load({manifest: versions(), currentVersion, page});
        const link = main.children[0].children[0];
        assert.equal(link.href, "https://example.test/EconML/_autosummary/econml.dml.LinearDML.html?q=forest#fit");
        context.window.location.href = context.window.location.href.replace("#fit", "#effect");
        const event = clickEvent();
        await link.callback(event);
        assert.equal(event.defaultPrevented, true);
        assert.deepEqual(probes, ["https://example.test/EconML/_autosummary/econml.dml.LinearDML.html?q=forest"]);
        assert.equal(context.navigated,
            "https://example.test/EconML/_autosummary/econml.dml.LinearDML.html?q=forest#effect");
    });
}

test("a missing stable page falls back to the stable home, not a 404 or obsolete anchor", async () => {
    const {main, context} = await load({
        manifest: versions(), currentVersion: "prerelease", page: "prerelease/future.html#new", headStatus: 404,
    });
    await main.children[0].children[0].callback(clickEvent());
    assert.equal(context.navigated, "https://example.test/EconML/index.html");
});

for (const options of [{headStatus: 500}, {headReject: true}]) {
    test(`stable-banner navigation reports an unexpected failure without leaving: ${JSON.stringify(options)}`, async () => {
        const {main, context, errors} = await load({manifest: versions(), ...options});
        const [link, status] = main.children[0].children;
        await link.callback(clickEvent());
        assert.equal(context.navigated, undefined);
        assert.equal(status.attributes.role, "status");
        assert.match(status.textContent, /Could not check the stable documentation\. Try again/);
        assert.equal(errors.length, 1);
    });
}

test("modified banner clicks retain native new-tab behavior with the corresponding page as the href", async () => {
    for (const options of [{ctrlKey: true}, {metaKey: true}, {shiftKey: true}, {altKey: true}, {button: 1}]) {
        const {main, probes, context} = await load({manifest: versions()});
        const link = main.children[0].children[0];
        const event = clickEvent(options);
        await link.callback(event);
        assert.equal(event.defaultPrevented, undefined);
        assert.equal(context.navigated, undefined);
        assert.equal(probes.length, 0);
        assert.equal(link.href, "https://example.test/EconML/_autosummary/econml.dml.LinearDML.html#fit");
    }
});

test("the prerelease warning and stable link remain usable if the manifest cannot load", async () => {
    const {main, select, context} = await load({
        manifest: versions(), currentVersion: "prerelease",
        page: "prerelease/_autosummary/econml.dml.LinearDML.html#fit", reject: true,
    });
    assert.equal(select.hidden, true);
    assert.match(main.children[0].textContent, /not yet released/);
    await main.children[0].children[0].callback(clickEvent());
    assert.equal(context.navigated, "https://example.test/EconML/_autosummary/econml.dml.LinearDML.html#fit");
});

test("an unrecognized theme retains an inline working picker and static fallback", async () => {
    const {panel, sidebar, select} = await load({manifest: versions(), themed: false});
    assert.equal(panel.parentElement, undefined);
    assert.equal(sidebar.children.length, 0);
    assert.equal(select.hidden, false);
    assert.equal(panel.querySelector("a").tagName, "a");
});

test("labels are text, not injected markup", async () => {
    const manifest = versions();
    manifest.versions[0].label = "<img src=x onerror=alert(1)>";
    const {select} = await load({manifest});
    assert.equal(select.options[0].textContent, manifest.versions[0].label);
    assert.equal(select.options[0].innerHTML, undefined);
});

for (const failure of ["http", "network", "schema", "path", "id", "duplicate", "missing", "stable"]) {
    test(`${failure} fails visibly and keeps the static fallback`, async () => {
        const manifest = versions();
        if (failure === "schema") { manifest.schema = 99; }
        if (failure === "path") { manifest.versions[0].path = "https://evil.test/"; }
        if (failure === "id") { manifest.versions[0].id = "../../outside"; }
        if (failure === "duplicate") { manifest.versions.push(manifest.versions[0]); }
        if (failure === "missing") { manifest.versions.pop(); }
        if (failure === "stable") { manifest.stable = "prerelease"; }
        const {select, status, errors} = await load({
            manifest, responseOK: failure !== "http", reject: failure === "network",
        });
        assert.equal(select.hidden, true);
        assert.equal(select.options.length, 0);
        assert.match(status.textContent, /all-versions link/);
        assert.equal(errors.length, 1);
    });
}
