// Start writing your queries here.
//
// You can use the schema to help you write your queries.
//
// Queries take the form:
//     QUERY {query name}({input name}: {input type}) =>
//         {variable} <- {traversal}
//         RETURN {variable}
//
// Example:
//     QUERY GetUserFriends(user_id: String) =>
//         friends <- N<User>(user_id)::Out<Knows>
//         RETURN friends
//
//
// For more information on how to write queries,
// see the documentation at https://docs.helix-db.com
// or checkout our GitHub at https://github.com/HelixDB/helix-db

// Insert file with embedding
QUERY insert_file(name: String, path: String, content: String, size: I64, vector: [F64]) =>
    file <- AddN<File>({name: name, path: path, content: content, size: size})
    embedding <- AddV<FileEmbedding>(vector, {content: content})
    AddE<HasEmbedding>::From(file)::To(embedding)
    RETURN file

// Search files by content similarity
QUERY search_files(query_vector: [F64], k: I64) =>
    embeddings <- SearchV<FileEmbedding>(query_vector, k)
    files <- embeddings::In<HasEmbedding>
    RETURN files

// Search by filename
QUERY search_by_name(filename: String) =>
    files <- SearchBM25<File>(filename, 10)
    RETURN files
