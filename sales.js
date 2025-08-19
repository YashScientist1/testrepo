const sales = [
    { item: "Laptop", quantity: 2, price: 800 },
    { item: "Monitor", quantity: 1, price: 150 },
    { item: "Mouse", quantity: 4, price: 25 }
]



function calculate_total_sales(sales) {

    let total=0;

    for (let i=0;i< sales.length; i++){

        total += sales[i].quantity * sales[i].price ;



    }

    return total;

}

console.log("total sles amount",calculate_total_sales(sales) );

