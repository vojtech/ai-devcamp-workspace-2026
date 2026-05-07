from mcp.server.fastmcp import FastMCP
import re
import requests

mcp = FastMCP("Health-Service")

storage = {
    "steps": 0
}



@mcp.tool()
def calculate_calories_burned(steps: int, weight_kg: float = 70.0) -> str:
    """Estimate kilocalories burned from walking a given number of steps.

    Uses the standard MET-based formula:
      kcal = steps × 0.00035 × weight_kg
    For an average 70 kg adult this gives ~300 kcal per 10 000 steps.
    """
    if steps <= 0:
        return "Please provide a positive number of steps."
    kcal = round(steps * 0.00035 * weight_kg)
    return (
        f"Walking {steps:,} steps burns approximately {kcal} kcal "
        f"(based on {weight_kg} kg body weight)."
    )


@mcp.tool()
def get_calories(food: str) -> str:
    """Fetch calorie data from Open Food Facts."""
    import time

    food_name = food.lower().strip()

    for attempt in range(3):
        try:
            response = requests.get(
                "https://world.openfoodfacts.org/cgi/search.pl",
                params={
                    "search_terms": food_name,
                    "search_simple": 1,
                    "action": "process",
                    "json": 1,
                    "page_size": 30,
                    "fields": "product_name,generic_name,nutriments",
                },
                timeout=15,
                headers={"User-Agent": "HealthFoodAgent/1.0 (educational project)"},
            )

            if response.status_code == 503:
                if attempt < 2:
                    time.sleep(1)
                    continue
                return f"Open Food Facts is temporarily unavailable (HTTP 503). Please try again in a moment."

            if not response.ok:
                return f"Open Food Facts returned an error (HTTP {response.status_code}) for '{food}'."

            if not response.text.strip():
                return f"Open Food Facts returned an empty response for '{food}'. Try again shortly."

            data = response.json()
            products = data.get("products", [])

            if not products:
                return f"No products found on Open Food Facts for '{food}'."

            checked = []
            for product in products:
                product_name = product.get("product_name") or ""
                generic_name = product.get("generic_name") or ""
                name = product_name or generic_name or food

                nutriments = product.get("nutriments", {})
                kcal = (
                    nutriments.get("energy-kcal_100g")
                    or nutriments.get("energy-kcal")
                    or nutriments.get("energy-kcal_value")
                )
                checked.append(name)

                if kcal is not None:
                    return f"{name} contains approximately {kcal} kcal per 100g."

            return (
                f"Found products for '{food}' on Open Food Facts, but none had calorie data. "
                f"Checked: {', '.join(checked[:5])}"
            )

        except Exception as error:
            if attempt < 2:
                time.sleep(1)
            else:
                return f"Could not fetch calorie data from Open Food Facts: {error}"

    return f"Open Food Facts is temporarily unavailable. Please try again in a moment."


def _lookup_kcal(food_name: str) -> float | None:
    """Return kcal-per-100g for a food, or None if not found."""
    try:
        r = requests.get(
            "https://world.openfoodfacts.org/cgi/search.pl",
            params={
                "search_terms": food_name.lower().strip(),
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": 10,
                "fields": "product_name,nutriments",
            },
            timeout=10,
            headers={"User-Agent": "HealthFoodAgent/1.0 (educational project)"},
        )
        if not r.ok or not r.text.strip():
            return None
        for product in r.json().get("products", []):
            n = product.get("nutriments", {})
            kcal = (
                n.get("energy-kcal_100g")
                or n.get("energy-kcal")
                or n.get("energy-kcal_value")
            )
            if kcal is not None:
                return float(kcal)
    except Exception:
        pass
    return None


@mcp.tool()
def get_calories_batch(foods: list[str]) -> str:
    """Look up kcal-per-100g for a list of food/ingredient names in one call.

    Returns a formatted breakdown of each ingredient's calorie density and the
    combined total (assuming 100 g of each). Skip items with no data.
    """
    rows = []
    total = 0.0
    skipped = []

    for food in foods:
        kcal = _lookup_kcal(food)
        if kcal is not None:
            rows.append(f"  {food}: {kcal:.0f} kcal / 100g")
            total += kcal
        else:
            skipped.append(food)

    if not rows:
        return (
            "Could not find calorie data for any of the provided ingredients. "
            "Open Food Facts may be temporarily unavailable."
        )

    lines = ["Calorie breakdown (per 100 g of each ingredient):"]
    lines.extend(rows)
    lines.append(f"\nCombined total (100 g each): {total:.0f} kcal")
    if skipped:
        lines.append(
            f"No data found for: {', '.join(skipped)} "
            "(these are likely spices or sauces with negligible calories)."
        )
    return "\n".join(lines)


@mcp.tool()
def get_ingredients(meal_name: str) -> str:
    """Fetch the ingredient list for a specific named meal from TheMealDB."""
    try:
        response = requests.get(
            "https://www.themealdb.com/api/json/v1/1/search.php",
            params={"s": meal_name},
            timeout=10,
        ).json()

        meals = response.get("meals")
        if not meals:
            return f"Could not find a meal called '{meal_name}' on TheMealDB."

        meal = meals[0]
        name = meal.get("strMeal", meal_name)

        ingredients = []
        for i in range(1, 21):
            ingredient = (meal.get(f"strIngredient{i}") or "").strip()
            measure = (meal.get(f"strMeasure{i}") or "").strip()
            if ingredient:
                ingredients.append(f"{measure} {ingredient}".strip() if measure else ingredient)

        if not ingredients:
            return f"Found '{name}' but it had no ingredient data."

        return f"Ingredients for {name}: " + ", ".join(ingredients) + "."

    except Exception as error:
        return f"Could not fetch ingredients from TheMealDB: {error}"


def _recipe_search(term: str):
    """Search TheMealDB by ingredient filter then by name. Returns (label, meals) or None."""
    try:
        r = requests.get(
            "https://www.themealdb.com/api/json/v1/1/filter.php",
            params={"i": term},
            timeout=10,
        ).json()
        if r.get("meals"):
            return term, r["meals"][:3]
    except Exception:
        pass

    try:
        r = requests.get(
            "https://www.themealdb.com/api/json/v1/1/search.php",
            params={"s": term},
            timeout=10,
        ).json()
        if r.get("meals"):
            return term, r["meals"][:3]
    except Exception:
        pass

    return None


@mcp.tool()
def get_recipe(ingredient: str, cuisine: str = "") -> str:
    """Fetch recipe ideas from TheMealDB."""
    cuisine = cuisine.strip()

    # Split compound queries like "chicken & mushroom hotpot" → ["chicken", "mushroom hotpot"]
    raw = ingredient.lower().strip()
    tokens = [t.strip() for t in re.split(r"[&,]|\band\b", raw) if t.strip()]
    candidates = [raw] + [t for t in tokens if t != raw]

    for candidate in candidates:
        result = _recipe_search(candidate)
        if result:
            label, meals = result
            names = ", ".join(meal["strMeal"] for meal in meals)
            return f"Recipe ideas using {label}: {names}"

    # Try cuisine search if given
    if cuisine:
        try:
            r = requests.get(
                "https://www.themealdb.com/api/json/v1/1/filter.php",
                params={"a": cuisine},
                timeout=10,
            ).json()
            if r.get("meals"):
                meals = r["meals"][:3]
                names = ", ".join(meal["strMeal"] for meal in meals)
                return (
                    f"Could not find an exact recipe for '{raw}', "
                    f"but here are {cuisine} recipe ideas: {names}"
                )
        except Exception:
            pass

    return (
        f"TheMealDB could not find a recipe for '{raw}'. "
        "Try simpler ingredient names like chicken, egg, rice, beef, pasta, salmon, tomato, or cheese."
    )


@mcp.tool()
def manage_steps(action: str, value: int = 0) -> str:
    """Manage step count. action can be add, get, or reset."""

    action = action.lower().strip()

    if action == "add":
        if value <= 0:
            return "Please provide a positive number of steps."

        storage["steps"] += value
        return f"Added {value} steps. Current total: {storage['steps']} steps."

    if action == "get":
        return f"Current step total: {storage['steps']} steps."

    if action == "reset":
        storage["steps"] = 0
        return "Step count reset to 0."

    return "Invalid action. Use add, get, or reset."


if __name__ == "__main__":
    mcp.run()